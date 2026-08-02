import os
import datetime
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.distributed.nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.util import *
from utils.logger import *
from utils.parser import get_clip_args
from utils.loss import clip_loss
from models.ECG_CLIP import ECGCLIP

from datasets.dataset_pretrain import ECGDataset

torch.autograd.set_detect_anomaly(True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_func(args):
    init_distributed_mode(args)
    
    # seed for random, numpy, torch, python, monai
    fix_random_seeds(args.seed)

    # logger 
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    print_log("==================== Parameters ====================", logger=logger)
    log_args_to_file(args, 'args', logger=logger)

    print_log("==================== preparing data ... ====================", logger=logger)

    train_ds, valid_ds = ECGDataset.create_datasets(os.path.join(args.data_root, args.h5_file_path), 
                                                    os.path.join(args.data_root, args.label_file_path), 
                                                    os.path.join(args.data_root, args.info_file_path), 
                                                    os.path.join(args.data_root, args.train_indices_file_path), 
                                                    os.path.join(args.data_root, args.valid_indices_file_path),
                                                    args.num_leads,
                                                    )

    print_log(f"==================== train dataset load complete ====================: {len(train_ds)} image-text pairs", logger=logger)
    print_log(f"==================== valid dataset load complete ====================: {len(valid_ds)} image-text pairs", logger=logger)

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        sampler=torch.utils.data.DistributedSampler(train_ds, shuffle=True),
        batch_size=args.train_batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        sampler=torch.utils.data.DistributedSampler(valid_ds, shuffle=False),
        batch_size=args.valid_batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print_log("==================== building network ... ====================", logger=logger)

    model = ECGCLIP(args)
    # move networks to gpu
    model = model.cuda()

    '''
    you can freeze bert from last layer to first layer.
    set the number of frozen layers with --text_encoder_freeze_layers
    default is freeze 9 layers
    '''
    if not args.freeze_text_encoder:
        if args.text_encoder_freeze_layers is not None:
            for layer_idx in range(int(args.text_encoder_freeze_layers)):
                for param in list(model.lm_model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
    else:
        for param in list(model.lm_model.parameters()):
            param.requires_grad = False

    # synchronize batch norm (if any)
    if has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # DDP wrapper...
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[args.gpu])

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    print_log("==================== preparing optimizer and schedulers ====================", logger=logger)

    update_params = get_params(model)
    params_groups = [{'params': update_params['params'], 'lr': args.lr}]

    optimizer = torch.optim.AdamW(
        params_groups,
        betas=(args.momentum, 0.999),
        weight_decay=args.weight_decay
    )

    # scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5000,
        T_mult=1,
        eta_min=1e-8,
    )

    print_log("Loss, optimizer and schedulers ready.", logger=logger)

    to_restore = {"epoch": 0}
    if args.resume:
        print_log("============ optionally resume training ... ============", logger=logger)
        restart_from_checkpoint(
            args.resume,
            run_variables=to_restore,
            model=model,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
        )

    start_epoch = to_restore["epoch"]

    print_log("Starting training !", logger=logger)

    writer = create_tfboard_on_master(os.path.join(args.tfboard_path))

    best_valid_stats = {'best_loss_epoch': -1,
                        'valid_loss': np.inf,
                        'best_acc_epoch': -1,
                        'valid_acc': -np.inf,
                        }

    for epoch in range(start_epoch, args.epochs):
        train_stats, valid_stats = train_valid_one_epoch(model,
                                                        train_dl,
                                                        valid_dl,
                                                        epoch,
                                                        best_valid_stats,
                                                        optimizer,
                                                        lr_scheduler,
                                                        fp16_scaler,
                                                        logger,
                                                        args,
                                                        writer,
                                                        )

        valid_loss = valid_stats['loss']
        valid_acc = valid_stats['acc1']

        if valid_loss < best_valid_stats['valid_loss']:
            best_valid_stats['valid_loss'] = valid_loss
            best_valid_stats['best_loss_epoch'] = epoch

        if valid_acc > best_valid_stats['valid_acc']:
            best_valid_stats['valid_acc'] = valid_acc
            best_valid_stats['best_acc_epoch'] = epoch


def train_one_epoch(model, data_loader, optimizer, lr_schedule, epoch, fp16_scaler, logger, args, writer):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    for it, subjects_batch in enumerate(metric_logger.log_every(data_loader, 10, header, logger)):
        # move tensor to gpu
        ecg, report = subjects_batch['ecg_data'].cuda(non_blocking=True), subjects_batch['report']

        with torch.cuda.amp.autocast(enabled=fp16_scaler is not None, dtype=torch.bfloat16):
            report_tokenize_output = model.module._tokenize(report)
            input_ids = report_tokenize_output.input_ids.cuda(non_blocking=True)
            attention_mask = report_tokenize_output.attention_mask.cuda(non_blocking=True)
            output_dict = model(ecg, input_ids, attention_mask) 
            ecg_emb, proj_ecg_emb, proj_text_emb = output_dict['ecg_emb'], output_dict['proj_ecg_emb'], output_dict['proj_text_emb']
            ecg_emb_1, ecg_emb_2 = ecg_emb

            # gather all tensors from gpus
            agg_ecg_emb_1, agg_ecg_emb_2 = distributed_concat(ecg_emb_1), distributed_concat(ecg_emb_2)
            agg_proj_ecg_emb = distributed_concat(proj_ecg_emb)
            agg_proj_text_emb = distributed_concat(proj_text_emb)

            cma_loss, acc1, acc5 = clip_loss(agg_proj_ecg_emb, agg_proj_text_emb)
            uma_loss, _, _ = clip_loss(agg_ecg_emb_1, agg_ecg_emb_2)
            loss = cma_loss + args.uma_weight * uma_loss
            res_dict = {
                'loss_cma': cma_loss,
                'loss_uma': uma_loss,
                'loss': loss,
                'acc1': acc1,
                'acc5': acc5,
            }

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        global_steps = epoch * len(data_loader) + it

        if is_main_process():
            # tensorboard
            for k, v in res_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                writer.add_scalar(f"train/{k}", v, global_steps)

        # logging
        lr_schedule.step()
        metric_logger.update(loss_train=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    train_status = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return train_status


def valid_one_epoch(model, data_loader, epoch, writer, logger, args):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    aggregated_res_dict = {
        'loss_cma': 0.0,
        'loss_uma': 0.0,
        'loss': 0.0,
        'acc1': 0.0,
        'acc5': 0.0,
    }
    count = 0

    # get feats from all gpus
    for it, subjects_batch in enumerate(metric_logger.log_every(data_loader, 10, header, logger)):
        ecg, report = subjects_batch['ecg_data'].cuda(non_blocking=True), subjects_batch['report']
        report_tokenize_output = model.module._tokenize(report)
        input_ids = report_tokenize_output.input_ids.cuda(non_blocking=True)
        attention_mask = report_tokenize_output.attention_mask.cuda(non_blocking=True)
        output_dict = model(ecg, input_ids, attention_mask) 
        ecg_emb, proj_ecg_emb, proj_text_emb = output_dict['ecg_emb'], output_dict['proj_ecg_emb'], output_dict['proj_text_emb']
        ecg_emb_1, ecg_emb_2 = ecg_emb

        # gather all tensors from gpus
        agg_ecg_emb_1, agg_ecg_emb_2 = distributed_concat(ecg_emb_1), distributed_concat(ecg_emb_2)
        agg_proj_ecg_emb = distributed_concat(proj_ecg_emb)
        agg_proj_text_emb = distributed_concat(proj_text_emb)

        cma_loss, acc1, acc5 = clip_loss(agg_proj_ecg_emb, agg_proj_text_emb)
        uma_loss, _, _ = clip_loss(agg_ecg_emb_1, agg_ecg_emb_2)
        loss = cma_loss + args.uma_weight * uma_loss
        res_dict = {
            'loss_cma': cma_loss.item(),
            'loss_uma': uma_loss.item(),
            'loss': loss.item(),
            'acc1': acc1,
            'acc5': acc5,
        }

        # Aggregate results
        for k, v in res_dict.items():
            aggregated_res_dict[k] += v
        count += 1

    # Average results over the dataset
    for k in aggregated_res_dict:
        aggregated_res_dict[k] /= count

    # Reduce dict
    reduced_res_dict = reduce_dict(aggregated_res_dict)

    if is_main_process():
        # tensorboard
        for k, v in reduced_res_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            writer.add_scalar(f"valid/{k}", v, epoch)

    return reduced_res_dict


def train_valid_one_epoch(model,
                          train_loader,
                          valid_loader,
                          epoch,
                          best_valid_stats,
                          optimizer,
                          lr_schedule,
                          fp16_scaler,
                          logger,
                          args,
                          writer,
                          ):
    start_time = time.time()
    # train
    model.train()
    train_stats = train_one_epoch(model,
                                  train_loader,
                                  optimizer,
                                  lr_schedule,
                                  epoch,
                                  fp16_scaler,
                                  logger,
                                  args,
                                  writer)

    # valid
    model.eval()
    with torch.no_grad():
        valid_stats = valid_one_epoch(
            model=model,
            data_loader=valid_loader,
            epoch=epoch,
            writer=writer,
            logger=logger,
            args=args,
        )
        valid_acc = valid_stats['acc1']
        valid_loss = valid_stats['loss']

    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'args': args,
    }
    if fp16_scaler is not None:
        save_dict['fp16_scaler'] = fp16_scaler.state_dict()

    save_on_master(save_dict, os.path.join(args.experiment_path, f'{args.exp_name}-checkpoint.pth'))

    pre_best_valid_loss, pre_best_valid_loss_epoch = best_valid_stats['valid_loss'], best_valid_stats['best_loss_epoch']
    pre_best_valid_acc, pre_best_valid_acc_epoch = best_valid_stats['valid_acc'], best_valid_stats['best_acc_epoch']
    if valid_loss < pre_best_valid_loss:
        if epoch > 0:
            filename_to_remove = os.path.join(args.experiment_path, f"{args.exp_name}-model-epoch{pre_best_valid_loss_epoch}-{pre_best_valid_loss:.2f}_loss.pt")
            remove_on_master(filename_to_remove)
        filename = os.path.join(args.experiment_path, f"{args.exp_name}-model-epoch{epoch}-{valid_loss:.2f}_loss.pt")
        save_on_master(save_dict, filename)

    if valid_acc > pre_best_valid_acc:
        if epoch > 0:
            filename_to_remove = os.path.join(args.experiment_path, f"{args.exp_name}-model-epoch{pre_best_valid_acc_epoch}-{pre_best_valid_acc:.2f}_acc.pt")
            remove_on_master(filename_to_remove)
        filename = os.path.join(args.experiment_path, f"{args.exp_name}-model-epoch{epoch}-{valid_acc:.2f}_acc.pt")
        save_on_master(save_dict, filename)

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 **{f'valid_{k}': v for k, v in valid_stats.items()},
                 'epoch': epoch}
    if is_main_process():
        log_loss_to_file(log_stats, logger=logger)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print_log('Training ' + ' time {}'.format(total_time_str))

    return train_stats, valid_stats


if __name__ == '__main__':
    args = get_clip_args()
    train_func(args)
