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

from datasets.dataset_downstream import ECGDatasetDownstream
from models.finetuner import FineTuner, get_backbone


def update_and_record_best_metrics(avg_metrics, avg_best_metrics, val_res_dict, best_metrics, writer, epoch):
    improved_metrics = []
    
    # 定义每个指标的比较方式：True表示越大越好，False表示越小越好
    comparison = {
        "loss": False,
        # "recall": True,
        "mcc": True,
        "precision": True,
        "f1_score": True,
        "pr_auc": True,
        "roc_auc": True,
        "sensitivity": True,
        "specificity": True,
        "mean_squared_error": False,  # MSE越小越好
        "root_mean_squared_error": False,  # RMSE越小越好
        "mean_absolute_error": False,  # MAE越小越好
        "r2_score": True  # R²越大越好
    }
    
    # 更新平均指标
    for metric in avg_metrics:
        if metric != 'roc_auc_classes':
            if ((not comparison.get(metric, True) and avg_metrics[metric] < avg_best_metrics[metric]) or 
                (comparison.get(metric, True) and avg_metrics[metric] > avg_best_metrics[metric])):
                avg_best_metrics[metric] = avg_metrics[metric]
                improved_metrics.append(metric)

    # 更新任务级指标
    for metric in improved_metrics:
        for task in val_res_dict:
            best_metrics[task][metric] = val_res_dict[task][metric]
            writer.add_scalar(f"valid/{task}/best_performance/{metric}", best_metrics[task][metric], epoch)
        
        # 记录平均性能指标
        writer.add_scalar(f"valid/average/best_performance/{metric}", avg_best_metrics[metric], epoch)


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

    # dataset & dataloader

    train_ds, valid_ds, test_ds = ECGDatasetDownstream.create_datasets(
                                        h5_file=os.path.join(args.data_root, args.h5_file_path), 
                                        label_file=os.path.join(args.data_root, args.label_file_path), 
                                        info_file=os.path.join(args.data_root, args.info_file_path), 
                                        train_index_file=os.path.join(args.data_root, args.train_indices_file_path), 
                                        valid_index_file=os.path.join(args.data_root, args.valid_indices_file_path),
                                        test_index_file=os.path.join(args.data_root, args.test_indices_file_path),
                                        task=args.downstream_task,
                                        ecg_or_echo=args.ecg_or_echo,
                                        exclude_label_ids=args.exclude_label_ids,
                                        random_split_seed=args.random_split_seed,
                                        sample_ratio=args.sample_ratio,
                                        num_leads=args.num_leads,
                                        downsample=args.downsample,
                                        )

    # handle class imbalance
    sample_weights = train_ds.get_sample_weights()
    sampler = None

    if sample_weights is not None:
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    if args.downstream_task == 'age':
        # criterion = SmoothL1Loss(weight=1.0, reduction='mean')
        criterion = nn.MSELoss(reduction='mean').cuda()
    else:
        # ce weight
        ce_weight = train_ds.get_class_weights()
        criterion = nn.CrossEntropyLoss(weight=ce_weight).cuda()

    train_dataloader = DataLoader(
        train_ds,
        pin_memory=True,
        drop_last=False,
        sampler=sampler if sampler is not None else None,
        shuffle=sampler is None,  # 如果没有使用采样器，则启用shuffle
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    valid_dataloader = DataLoader(
        valid_ds,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print_log(f"==================== train dataset load complete ====================: {len(train_ds)} image-text pairs", logger=logger)
    print_log(f"==================== valid dataset load complete ====================: {len(valid_ds)} image-text pairs", logger=logger)

    train_label_distribution = train_ds.get_task_info()
    valid_label_distribution = valid_ds.get_task_info()
    log_dict_to_logger(train_label_distribution, pre="train data distribution", logger=logger)
    log_dict_to_logger(valid_label_distribution, pre="valid data distribution", logger=logger)

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
        multi_task_ids = train_ds.get_label_ids()
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
    elif args.ecg_model == 'regnet':
        in_features = 1024
        backbone_name = 'regnet'
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
    print_log("==================== optimizer ====================", logger=logger)

    # optimizer
    update_params = get_params(finetuner)
    params_groups = [{'params': update_params['params'], 'lr': args.lr}]
    optimizer = torch.optim.AdamW(
        params_groups,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    lr_scheduler =  CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=args.total_iterations,
            cycle_mult=1.0,
            max_lr=(args.lr, ),
            min_lr=(1e-6, ),
            warmup_steps=int(0.05 * args.total_iterations),
            gamma=1.0,
        )

    # metrics
    print_log("==================== metric ====================", logger=logger)

    # 初始化best_metrics和avg_best_metrics
    best_metrics = {}
    best_thresholds_dict = {}
    avg_best_metrics = {
        "loss": float('inf'),
    }

    # 一些示例：根据任务名称来判断任务类型 (可以根据实际情况调整)
    classification_tasks = {"gender", "ef", "af", "MS", "MR", "AS", "AR", "TS", "TR", "PS", "PR"}
    regression_tasks = {"age"}

    for task_name in task_info:
        if task_name in regression_tasks:
            best_metrics[task_name] = {
                "loss": float('inf'),
                "mean_squared_error": float('inf'),
                "root_mean_squared_error": float('inf'),
                "mean_absolute_error": float('inf'),
                "r2_score": 0.0,
            }
            avg_best_metrics.update({
                "mean_squared_error": float('inf'),
                "root_mean_squared_error": float('inf'),
                "mean_absolute_error": float('inf'),
                "r2_score": 0.0,
            })
        else:  # 如果不是回归任务，则默认为分类任务
            best_metrics[task_name] = {
                "loss": float('inf'),
                "sensitivity": 0.0,
                # "precision": 0.0,
                "f1_score": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "specificity": 0.0,
                "mcc": -1.0,
            }
            avg_best_metrics.update({
                "sensitivity": 0.0,
                # "precision": 0.0,
                "f1_score": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "specificity": 0.0,
                "mcc": -1.0,
            })

    # 去重后保留最终的avg_best_metrics初始化
    avg_best_metrics = {k: v for k, v in avg_best_metrics.items()}

    # tensorboard
    writer = create_tfboard_on_master(os.path.join(args.tfboard_path))

    if not args.test_only:
        print_log("==================== start training ====================", logger=logger)
        global_step = 0  # 初始化迭代数量

        metric_logger = MetricLogger(delimiter="  ")
        header = 'Training Iterations'
        finetuner.train()

        while global_step < args.total_iterations:
            for it, batch in enumerate(metric_logger.log_every(train_dataloader, 100, header, logger)):
                if global_step >= args.total_iterations:
                    break

                batch['ecg_data'] = batch['ecg_data'].cuda()

                # Check if y is a dictionary (multi-task situation) and handle appropriately
                if isinstance(batch['label'], dict):
                    for key in batch['label']:
                        batch['label'][key] = batch['label'][key].cuda().squeeze()
                else:
                    batch['label'] = batch['label'].cuda().squeeze()

                y = batch['label']
                logits = finetuner(batch['ecg_data'])

                task_losses = {
                    task: criterion(logits[task].float(), y[task].float() if task == 'age' else y[task].long())
                    for task in task_info.keys()
                }

                # 计算总损失，假设在多任务情况下，损失是每个任务损失的总和
                total_loss = sum(task_losses.values())

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                res_dict = get_metrics(logits_dict=logits, labels_dict=y, task_info=task_info, mode='train')

                # Initialize average metrics based on task type
                if args.downstream_task == 'age':
                    avg_metrics = {
                        "loss": 0.0,
                        "mean_squared_error": 0.0,
                        "root_mean_squared_error": 0.0,
                        "mean_absolute_error": 0.0,
                        "r2_score": 0.0,
                    }
                else:
                    avg_metrics = {
                        "loss": 0.0,
                        "sensitivity": 0.0,
                        # "precision": 0.0,
                        "f1_score": 0.0,
                        "pr_auc": 0.0,
                        "roc_auc": 0.0,
                        "specificity": 0.0,
                        "mcc": 0.0, 
                    }

                # 记录每个任务的metrics
                for task, metrics in res_dict.items():
                    task_loss = task_losses[task].item()
                    avg_metrics['loss'] += task_loss
                    writer.add_scalar(f"{task}/loss", task_loss, global_step)
                    for metric, value in metrics.items():
                        writer.add_scalar(f"{task}/{metric}", value, global_step)
                        # Accumulate metrics for average calculation
                        if metric in avg_metrics:
                            avg_metrics[metric] += value

                # 计算所有任务的平均metrics
                num_tasks = len(task_info)
                for metric in avg_metrics:
                    avg_metrics[metric] /= num_tasks

                # 记录所有任务的平均metrics
                for metric, value in avg_metrics.items():
                    writer.add_scalar(f"average/{metric}", value, global_step)

                log_loss_to_file(avg_metrics, logger=logger)
                metric_logger.update(loss_train=total_loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

                global_step += 1  # 增加迭代计数
                lr_scheduler.step()

                # 验证过程
                if global_step % args.val_iter == 0:
                    finetuner.eval()

                    logits_dict = {task: [] for task in task_info.keys()}
                    labels_dict = {task: [] for task in task_info.keys()}
                    loss_dict = {task: [] for task in task_info.keys()}
                    # pid_list = []
                    with torch.no_grad():
                        val_metric_logger = MetricLogger(delimiter="  ")
                        for it, batch in enumerate(metric_logger.log_every(valid_dataloader, 100, header, logger)):

                            pid = batch['filename']
                            batch['ecg_data'] = batch['ecg_data'].cuda()
                            if isinstance(batch['label'], dict):
                                for key in batch['label']:
                                    batch['label'][key] = batch['label'][key].cuda()
                            else:
                                batch['label'] = batch['label'].cuda()
                            logits = finetuner(batch['ecg_data'])

                            y = batch['label']

                            # Collect logits and labels
                            for task in task_info.keys():
                                task_logits = logits[task].float()
                                task_labels = y[task].float() if task == 'age' else y[task].long()  # Age task uses float labels

                                task_losses[task] = criterion(task_logits, task_labels)
                                logits_dict[task].append(task_logits.detach().cpu().numpy())
                                labels_dict[task].append(task_labels.detach().cpu().numpy())
                                loss_dict[task].append(task_losses[task].item())

                        # Concatenate collected results
                        for task in task_info.keys():
                            logits_dict[task] = np.concatenate(logits_dict[task], axis=0)
                            labels_dict[task] = np.concatenate(labels_dict[task], axis=0)
                            loss_dict[task] = np.mean(loss_dict[task])

                        # Calculate metrics
                        val_res_dict, best_thresholds_dict = get_metrics(logits_dict=logits_dict, labels_dict=labels_dict, task_info=task_info, mode='valid')
                        # Add per-task loss to val_res_dict
                        for task in val_res_dict.keys():
                            val_res_dict[task]["loss"] = loss_dict[task]

                        if args.downstream_task == 'age':
                            avg_metrics = {
                                "loss": 0.0,
                                "mean_squared_error": 0.0,
                                "root_mean_squared_error": 0.0,
                                "mean_absolute_error": 0.0,
                                "r2_score": 0.0,
                            }
                        else:
                            avg_metrics = {
                                "loss": 0.0,
                                "sensitivity": 0.0,
                                "f1_score": 0.0,
                                "pr_auc": 0.0,
                                "roc_auc": 0.0,
                                "specificity": 0.0,
                                "mcc": 0.0,
                            }

                        # Log metrics for each task
                        for task, metrics in val_res_dict.items():
                            for metric, value in metrics.items():
                                if isinstance(value, dict):
                                    # 如果值是字典，逐项记录
                                    for sub_metric, sub_value in value.items():
                                        writer.add_scalar(f"valid/{task}/{metric}/{sub_metric}", sub_value, global_step)
                                else:
                                    writer.add_scalar(f"valid/{task}/{metric}", value, global_step)

                                # Update average metrics
                                if metric in avg_metrics:
                                    avg_metrics[metric] += value


                        for task in val_res_dict.keys():
                            val_res_dict[task]["loss"] = loss_dict[task]
                            
                        # Calculate average of metrics
                        for metric in avg_metrics:
                            avg_metrics[metric] /= len(task_info)

                        # Log average metrics
                        for metric, value in avg_metrics.items():
                            writer.add_scalar(f"valid/average/{metric}", value, global_step)

                        # Save best ckpt
                        save_dict = {
                            'model': finetuner.state_dict(),
                            'threshold': best_thresholds_dict,
                        }
                        # if (args.downstream_task != 'age' and avg_metrics["roc_auc"] > avg_best_metrics["roc_auc"]) or \
                        # if (args.downstream_task != 'age' and avg_metrics["mcc"] > avg_best_metrics["mcc"]) or \
                        if (args.downstream_task != 'age' and avg_metrics["pr_auc"] > avg_best_metrics["pr_auc"]) or \
                           (args.downstream_task == 'age' and avg_metrics["mean_squared_error"] < avg_best_metrics["mean_squared_error"]):
                            torch.save(save_dict, os.path.join(args.experiment_path, 'best-checkpoint.pth'))

                        # Update and log average best metrics if any metric's average improved
                        update_and_record_best_metrics(avg_metrics, avg_best_metrics, val_res_dict, best_metrics, writer, global_step)
                        print("============================")

                    finetuner.train()

        # torch.save(save_dict, os.path.join(args.experiment_path, 'best-checkpoint.pth'))

    # test
    print_log("==================== start testing ====================", logger=logger)
    # load pre-trained img encoder
    checkpoint = torch.load(os.path.join(args.experiment_path, 'best-checkpoint.pth'))
    finetuner.load_state_dict(checkpoint['model'])
    finetuner.eval()
    best_thresholds_dict = checkpoint['threshold']

    # # internal validation
    # hospital = 'internal'
    # test_model(
    #         test_ds=test_ds,
    #         finetuner=finetuner,
    #         task_info=task_info,
    #         hospital=hospital,
    #         logger=logger,
    #         writer=writer,
    #         args=args,
    #         threshold=best_thresholds_dict,
    #         save_flag=True,
    #         )


    # validation inference: save validation logits for threshold calibration
    test_model(
        eval_ds=valid_ds,
        finetuner=finetuner,
        task_info=task_info,
        split_name='valid',
        logger=logger,
        writer=writer,
        args=args,
        threshold=best_thresholds_dict,
        save_flag=True,
    )

    # internal test inference: reuse the validation-derived threshold
    test_model(
        eval_ds=test_ds,
        finetuner=finetuner,
        task_info=task_info,
        split_name='internal_test',
        logger=logger,
        writer=writer,
        args=args,
        threshold=best_thresholds_dict,
        save_flag=True,
    )

# def test_model(test_ds, finetuner, task_info, hospital, logger, writer, args, threshold, save_flag=False):
#     test_dataloader = DataLoader(
#         test_ds,
#         pin_memory=True,
#         shuffle=False,
#         drop_last=False,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#     )
#     print_log(f"==================== {hospital} test dataset load complete ====================: {len(test_ds)} images", logger=logger)
#     test_label_distribution = test_ds.get_task_info()
#     log_dict_to_logger(test_label_distribution, pre=f"{hospital} test data distribution", logger=logger)

#     logits_dict = {task: [] for task in task_info.keys()}
#     labels_dict = {task: [] for task in task_info.keys()}
#     result_list = []

#     with torch.no_grad():
#         test_metric_logger = MetricLogger(delimiter="  ")
#         header = 'Test:'
        
#         for batch in test_metric_logger.log_every(test_dataloader, 10, header, logger):
#             pid = batch['filename']
#             batch['ecg_data'] = batch['ecg_data'].cuda()
            
#             if isinstance(batch['label'], dict):
#                 for key in batch['label']:
#                     batch['label'][key] = batch['label'][key].cuda()
#             else:
#                 batch['label'] = batch['label'].cuda()
            
#             logits = finetuner(batch['ecg_data'])
#             y = batch['label']
            
#             if isinstance(y, dict):
#                 for key in y:
#                     y[key] = y[key].squeeze()
#             else:
#                 y = y.squeeze()
            
#             for task in task_info.keys():
#                 task_logits = logits[task].float()
#                 task_labels = y[task].float() if task == 'age' else y[task].long()  # Age task uses float labels

#                 task_logits = task_logits.detach().cpu().numpy()
#                 task_labels = task_labels.detach().cpu().numpy()
                
#                 logits_dict[task].append(task_logits)
#                 labels_dict[task].append(task_labels)

#                 # 计算预测结果的正确与否
#                 # preds = torch.argmax(task_logits, dim=1)

#                 # 用传入的阈值计算指标
#                 if len(task_logits.shape) == 2:
#                     # 稳定 softmax 的计算
#                     task_logits = task_logits - np.max(task_logits, axis=1, keepdims=True)
#                     # 计算概率
#                     softmax_logits = np.exp(task_logits) / np.sum(np.exp(task_logits), axis=1, keepdims=True)
#                     predicted_probs = softmax_logits[:, 1]
#                 elif len(task_logits.shape) == 1:
#                     predicted_probs = task_logits
#                 best_threshold = threshold[task]
#                 preds = (predicted_probs >= best_threshold).astype(int)

#                 correct = preds == task_labels

#                 num_samples_in_batch = len(pid)
#                 task_name_list = [task] * num_samples_in_batch

#                 result_list.extend(zip(pid, task_name_list, task_labels, preds, correct, task_logits))
                        
#         for task in task_info.keys():
#             logits_dict[task] = np.concatenate(logits_dict[task], axis=0)
#             labels_dict[task] = np.concatenate(labels_dict[task], axis=0)
                
#         test_res_dict = get_metrics(logits_dict=logits_dict, labels_dict=labels_dict, task_info=task_info, mode='test', threshold=threshold)
        
#         avg_metrics = {metric: 0.0 for metric in test_res_dict[next(iter(test_res_dict))].keys()}
        
#         for task, metrics in test_res_dict.items():
#             for metric, value in metrics.items():
#                 if isinstance(value, dict):
#                     for sub_metric, sub_value in value.items():
#                         writer.add_scalar(f"test/{task}/{hospital}/{metric}/{sub_metric}", sub_value)
#                 else:
#                     writer.add_scalar(f"test/{task}/{hospital}/{metric}", value)
                
#                 if not isinstance(value, dict):
#                     avg_metrics[metric] += value
        
#         for metric in avg_metrics:
#             avg_metrics[metric] /= len(task_info)
        
#         for metric, value in avg_metrics.items():
#             writer.add_scalar(f"test/average/{hospital}/{metric}", value)

#         if save_flag:
#             # 将结果写入CSV文件
#             with open(os.path.join(args.logits_path, f'{hospital}_test_results.csv'), mode='w', newline='') as file:
#                 csv_writer  = csv.writer(file)
#                 # writer.writerow(['PID', 'Label', 'Pred Label', 'Correct', 'Logits'])
#                 csv_writer .writerow(['PID', 'Task', 'Label', 'Pred_Label', 'Correct', 'Logits'])
#                 csv_writer .writerows(result_list)


def test_model(eval_ds, finetuner, task_info, split_name, logger, writer, args, threshold, save_flag=False):
    eval_dataloader = DataLoader(
        eval_ds,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print_log(
        f"==================== {split_name} dataset load complete ====================: {len(eval_ds)} images",
        logger=logger
    )

    eval_label_distribution = eval_ds.get_task_info()
    log_dict_to_logger(eval_label_distribution, pre=f"{split_name} data distribution", logger=logger)

    logits_dict = {task: [] for task in task_info.keys()}
    labels_dict = {task: [] for task in task_info.keys()}
    result_list = []

    finetuner.eval()

    with torch.no_grad():
        eval_metric_logger = MetricLogger(delimiter="  ")
        header = f'{split_name}:'

        for batch in eval_metric_logger.log_every(eval_dataloader, 10, header, logger):
            pid = batch['filename']

            if args.cuda:
                batch['ecg_data'] = batch['ecg_data'].cuda()
            else:
                batch['ecg_data'] = batch['ecg_data']

            if isinstance(batch['label'], dict):
                for key in batch['label']:
                    if args.cuda:
                        batch['label'][key] = batch['label'][key].cuda()
                    else:
                        batch['label'][key] = batch['label'][key]
            else:
                if args.cuda:
                    batch['label'] = batch['label'].cuda()

            logits = finetuner(batch['ecg_data'])
            y = batch['label']

            if isinstance(y, dict):
                for key in y:
                    y[key] = y[key].squeeze()
            else:
                y = y.squeeze()

            for task in task_info.keys():
                task_logits_tensor = logits[task].float()
                task_labels_tensor = y[task].float() if task == 'age' else y[task].long()

                task_logits_raw = task_logits_tensor.detach().cpu().numpy()
                task_labels = task_labels_tensor.detach().cpu().numpy()

                logits_dict[task].append(task_logits_raw)
                labels_dict[task].append(task_labels)

                # Convert model logits to positive-class probability for thresholding.
                # For binary classification, task_logits_raw is usually [N, 2].
                if len(task_logits_raw.shape) == 2:
                    logits_stable = task_logits_raw - np.max(task_logits_raw, axis=1, keepdims=True)
                    exp_logits = np.exp(logits_stable)
                    softmax_logits = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                    predicted_probs = softmax_logits[:, 1]
                elif len(task_logits_raw.shape) == 1:
                    predicted_probs = task_logits_raw
                else:
                    raise ValueError(f"Unexpected logits shape for task {task}: {task_logits_raw.shape}")

                best_threshold = threshold[task]
                preds = (predicted_probs >= best_threshold).astype(int)

                correct = preds == task_labels

                num_samples_in_batch = len(pid)
                task_name_list = [task] * num_samples_in_batch

                # Save raw logits as Python lists so downstream ast.literal_eval can parse them.
                if len(task_logits_raw.shape) == 2:
                    logits_to_save = [x.tolist() for x in task_logits_raw]
                else:
                    logits_to_save = task_logits_raw.tolist()

                result_list.extend(
                    zip(
                        pid,
                        task_name_list,
                        task_labels.tolist(),
                        preds.tolist(),
                        correct.tolist(),
                        logits_to_save,
                    )
                )

        for task in task_info.keys():
            logits_dict[task] = np.concatenate(logits_dict[task], axis=0)
            labels_dict[task] = np.concatenate(labels_dict[task], axis=0)

        eval_res_dict = get_metrics(
            logits_dict=logits_dict,
            labels_dict=labels_dict,
            task_info=task_info,
            mode='test',
            threshold=threshold
        )

        avg_metrics = {metric: 0.0 for metric in eval_res_dict[next(iter(eval_res_dict))].keys()}

        for task, metrics in eval_res_dict.items():
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    for sub_metric, sub_value in value.items():
                        writer.add_scalar(f"{split_name}/{task}/{metric}/{sub_metric}", sub_value)
                else:
                    writer.add_scalar(f"{split_name}/{task}/{metric}", value)

                if not isinstance(value, dict):
                    avg_metrics[metric] += value

        for metric in avg_metrics:
            avg_metrics[metric] /= len(task_info)

        for metric, value in avg_metrics.items():
            writer.add_scalar(f"{split_name}/average/{metric}", value)

        if save_flag:
            output_csv = os.path.join(args.logits_path, f'{split_name}_results.csv')

            with open(output_csv, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(['PID', 'Task', 'Label', 'Pred_Label', 'Correct', 'Logits'])
                csv_writer.writerows(result_list)

            print_log(f"Saved logits to: {output_csv}", logger=logger)
            

if __name__ == '__main__':
    args = get_downstream_args()
    main(args)
