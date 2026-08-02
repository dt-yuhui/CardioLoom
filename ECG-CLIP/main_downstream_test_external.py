import os

import numpy as np
import torch
import time
import argparse
import sys
import csv
import torch.nn.functional as F

from utils.util import *
from utils.loss import SmoothL1Loss
from utils.logger import *
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.parser import get_downstream_args
import pandas as pd

from datasets.dataset_shiyuan import ShiyuanDataset
from datasets.dataset_ukbiobank import UKBioBankDataset
from datasets.dataset_xm import XiamenDataset
from datasets.dataset_mimiciv import MimicivDataset
from datasets.dataset_ptbxl import PtbxlDataset
from datasets.dataset_georgia import GeorgiaDataset
from datasets.dataset_cpsc2018 import Cpsc2018Dataset
from datasets.dataset_chapman import ChapmanDataset
from datasets.dataset_bidmc import BidmcDataset


from models.finetuner import FineTuner, get_backbone

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)

    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)
    else:
        # 清空 TFBoard 文件夹
        for filename in os.listdir(args.tfboard_path):
            file_path = os.path.join(args.tfboard_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print('Deleted file: %s' % file_path)

    # save logits
    args.logits_path = os.path.join(args.experiment_path, "logits")
    if not os.path.exists(args.logits_path):
        os.makedirs(args.logits_path)
        print('Create logits path successfully at %s' % args.logits_path)


def main(args):
    create_experiment_dir(args)
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    log_args_to_file(args, 'args', logger=logger)
    # 调用装饰器，动态传递logger_name和log_file
    decorated_main = log_exceptions(logger_name=args.log_name, log_file=log_file)(main_func)
    decorated_main(args, logger)


def main_func(args, logger):
    fix_random_seeds(args.seed)

    if args.external_dataset == 'shiyuan':
        test_ds = ShiyuanDataset(
            task=args.downstream_task,
            exclude_label_ids=args.exclude_label_ids,
            num_leads=args.num_leads,
            downsample=args.downsample,
            )
    elif args.external_dataset == 'bidmc':
        test_ds = BidmcDataset(
            task=args.downstream_task,
            exclude_label_ids=args.exclude_label_ids,
            num_leads=args.num_leads,
            downsample=args.downsample,
            )
    elif args.external_dataset == 'ukbiobank':
        test_ds = UKBioBankDataset(
            task=args.downstream_task,
            exclude_label_ids=args.exclude_label_ids,
            num_leads=args.num_leads,
            downsample=args.downsample,
        )
    elif args.external_dataset == 'xiamen':
        test_ds = XiamenDataset(
            task=args.downstream_task,
            exclude_label_ids=args.exclude_label_ids,
            num_leads=args.num_leads,
            downsample=args.downsample,
        )
    elif args.external_dataset == 'mimiciv':
        test_ds = MimicivDataset(
            task=args.downstream_task,
            exclude_label_ids=args.exclude_label_ids,
            num_leads=args.num_leads,
            downsample=args.downsample,
        )
    elif args.external_dataset == 'ptbxl':
        test_ds = PtbxlDataset(
            task=args.downstream_task,
            exclude_label_ids=args.exclude_label_ids,
            num_leads=args.num_leads,
            downsample=args.downsample,
        )
    elif args.external_dataset == 'georgia':
        test_ds = GeorgiaDataset(
            task=args.downstream_task,
            exclude_label_ids=args.exclude_label_ids,
            num_leads=args.num_leads,
            downsample=args.downsample,
        )
    elif args.external_dataset == 'cpsc2018':
        test_ds = Cpsc2018Dataset(
            task=args.downstream_task,
            exclude_label_ids=args.exclude_label_ids,
            num_leads=args.num_leads,
            downsample=args.downsample,
        )
    elif args.external_dataset == 'chapman':
        test_ds = ChapmanDataset(
            task=args.downstream_task,
            exclude_label_ids=args.exclude_label_ids,
            num_leads=args.num_leads,
            downsample=args.downsample,
        )           

    # device
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print_log('Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices', logger=logger)
    else:
        print_log('Using CPU', logger=logger)

    backbone, msg = get_backbone(args)
    print_log(f"ecg encoder from {args.pretrained_exp_name}: {msg}", logger=logger)
    device = torch.device("cuda" if args.cuda else "cpu")

    if "multi" not in args.downstream_task:
        task_info = {args.downstream_task: args.num_classes}
    else:
        multi_task_ids = test_ds.get_label_ids()
        task_info = {}
        for task_name in multi_task_ids:
            task_classes = 2
            task_info.update({task_name: task_classes})

    if 'resnet' in args.ecg_model:
        backbone_name = 'resnet'
        if args.ecg_model == 'resnet18' or args.ecg_model == 'resnet34':
            in_features = 512
        else:
            in_features = 2048
    elif args.ecg_model == 'ked':
        backbone_name = 'ked'
    elif args.ecg_model == 'merl':
        backbone_name = 'resnet'
        in_features = 512

    finetuner = FineTuner(backbone=backbone,
                          backbone_name=backbone_name,
                          in_features=in_features,
                          dropout=0,
                          finetune_type=args.finetune_type,
                          task_info=task_info,
                          )
    
    finetuner = finetuner.to(device)

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # metrics
    print_log("==================== metric ====================", logger=logger)

    # tensorboard
    writer = create_tfboard_on_master(os.path.join(args.tfboard_path))

    # test
    print_log("==================== start testing ====================", logger=logger)
    # load pre-trained img encoder
    checkpoint = torch.load(os.path.join(args.ckpt_path, 'best-checkpoint.pth'))
    # 确保返回值接收正确
    load_status = finetuner.load_state_dict(checkpoint['model'], strict=False)

    # 打印调试信息（如果需要）
    print_log(f"Missing keys: {load_status.missing_keys}", logger=logger)
    print_log(f"Unexpected keys: {load_status.unexpected_keys}", logger=logger)

    finetuner.eval()
    best_thresholds_dict = checkpoint['threshold']
    # remove specific tasks without positive samples
    if args.external_dataset == 'shiyuan':
        task_to_remove = ['echo_label_24']
    elif args.external_dataset == 'bidmc':
        existing_task_ids = [3, 5, 12, 15, 16, 17, 20, 22, 27]
        all_tasks = [f'echo_label_{i}' for i in range(1, 29)]
        task_to_remove = [task for task in all_tasks if int(task.split('_')[-1]) not in existing_task_ids]
    else:
        if args.external_dataset == 'ukbiobank':
            existing_task_ids = [1, 3, 4, 6, 8, 9, 11, 13, 14, 15, 16, 19, 27, 32, 33, 34, 35, 36, 38, 39, 41, 42, 44, 53, 57]
        elif args.external_dataset == 'xiamen':
            existing_task_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 18, 19, 21, 24, 26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 52, 53, 55, 57, 58]
        elif args.external_dataset == 'mimiciv':
            existing_task_ids = [1, 2, 3, 4, 6, 7, 8, 11, 13, 14, 15, 17, 18, 19, 21, 24, 27, 28, 29, 31, 32, 33, 34, 36, 38, 39, 40, 41, 42, 43, 44, 46, 49, 53, 55, 57, 58]
        elif args.external_dataset == 'ptbxl':
            existing_task_ids = [1, 2, 3, 4, 6, 7, 8, 9, 14, 15, 17, 18, 19, 27, 28, 29, 31, 32, 33, 34, 36, 38, 39, 41, 42, 44, 45, 46, 47, 48, 49, 51, 53]
        elif args.external_dataset == 'georgia':
            existing_task_ids = [2, 3, 4, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 24, 27, 28, 29, 31, 32, 33, 34, 36, 38, 39, 41, 42, 44, 47, 51, 53, 57]
        elif args.external_dataset == 'cpsc2018':
            existing_task_ids = [1, 6, 8, 19, 27, 32, 33, 44]
        elif args.external_dataset == 'chapman':
            existing_task_ids = [2, 3, 4, 6, 7, 8, 9, 13, 16, 17, 18, 19, 24, 27, 28, 29, 31, 33, 34, 36, 39, 41, 42, 44, 53, 56, 57]
        all_tasks = [f'ecg_label_{i}' for i in range(1, 74)]
        task_to_remove = [task for task in all_tasks if int(task.split('_')[-1]) not in existing_task_ids]
    external_dataset_task_info = {k: v for k, v in task_info.items() if k not in task_to_remove}

    # internal validation
    hospital = args.external_dataset

    test_model(
            test_ds=test_ds,
            finetuner=finetuner,
            task_info=external_dataset_task_info,
            hospital=hospital,
            logger=logger,
            writer=writer,
            args=args,
            threshold=best_thresholds_dict,
            save_flag=True,
            )

def test_model(test_ds, finetuner, task_info, hospital, logger, writer, args, threshold, save_flag=False):
    test_dataloader = DataLoader(
        test_ds,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print_log(f"==================== {hospital} test dataset load complete ====================: {len(test_ds)} images", logger=logger)
    test_label_distribution = test_ds.get_task_info()
    log_dict_to_logger(test_label_distribution, pre=f"{hospital} test data distribution", logger=logger)

    logits_dict = {task: [] for task in task_info.keys()}
    labels_dict = {task: [] for task in task_info.keys()}
    result_list = []

    with torch.no_grad():
        test_metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'
        
        for batch in test_metric_logger.log_every(test_dataloader, 10, header, logger):
            pid = batch['filename']
            batch['ecg_data'] = batch['ecg_data'].cuda()
            
            if isinstance(batch['label'], dict):
                for key in batch['label']:
                    batch['label'][key] = batch['label'][key].cuda()
            else:
                batch['label'] = batch['label'].cuda()
            
            logits = finetuner(batch['ecg_data'])
            y = batch['label']
            
            if isinstance(y, dict):
                for key in y:
                    y[key] = y[key].squeeze()
            else:
                y = y.squeeze()
            
            for task in task_info.keys():
                task_logits = logits[task].float()
                task_labels = y[task].float() if task == 'age' else y[task].long()  # Age task uses float labels

                task_logits = task_logits.detach().cpu().numpy()
                task_labels = task_labels.detach().cpu().numpy()
                
                logits_dict[task].append(task_logits)
                labels_dict[task].append(task_labels)

                # 计算预测结果的正确与否
                # preds = torch.argmax(task_logits, dim=1)

                # 用传入的阈值计算指标
                if len(task_logits.shape) == 2:
                    # 稳定 softmax 的计算
                    task_logits = task_logits - np.max(task_logits, axis=1, keepdims=True)
                    # 计算概率
                    softmax_logits = np.exp(task_logits) / np.sum(np.exp(task_logits), axis=1, keepdims=True)
                    predicted_probs = softmax_logits[:, 1]
                elif len(task_logits.shape) == 1:
                    predicted_probs = task_logits
                best_threshold = threshold[task]
                preds = (predicted_probs >= best_threshold).astype(int)

                correct = preds == task_labels

                num_samples_in_batch = len(pid)
                task_name_list = [task] * num_samples_in_batch

                result_list.extend(zip(pid, task_name_list, task_labels, preds, correct, task_logits))
                        
        for task in task_info.keys():
            logits_dict[task] = np.concatenate(logits_dict[task], axis=0)
            labels_dict[task] = np.concatenate(labels_dict[task], axis=0)
                
        test_res_dict = get_metrics(logits_dict=logits_dict, labels_dict=labels_dict, task_info=task_info, mode='test', threshold=threshold)
        
        avg_metrics = {metric: 0.0 for metric in test_res_dict[next(iter(test_res_dict))].keys()}
        
        for task, metrics in test_res_dict.items():
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    for sub_metric, sub_value in value.items():
                        writer.add_scalar(f"test/{task}/{hospital}/{metric}/{sub_metric}", sub_value)
                else:
                    writer.add_scalar(f"test/{task}/{hospital}/{metric}", value)
                
                if not isinstance(value, dict):
                    avg_metrics[metric] += value
        
        for metric in avg_metrics:
            avg_metrics[metric] /= len(task_info)
        
        for metric, value in avg_metrics.items():
            writer.add_scalar(f"test/average/{hospital}/{metric}", value)

        if save_flag:
            # 将结果写入CSV文件
            with open(os.path.join(args.logits_path, f'{hospital}_test_results.csv'), mode='w', newline='') as file:
                csv_writer  = csv.writer(file)
                # csv_writer.writerow(['PID', 'Label', 'Pred Label', 'Correct', 'Logits'])
                csv_writer .writerow(['PID', 'Task', 'Label', 'Pred_Label', 'Correct', 'Logits'])
                csv_writer .writerows(result_list)

if __name__ == '__main__':
    args = get_downstream_args()
    main(args)
