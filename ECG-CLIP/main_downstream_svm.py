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
import pickle
from scipy.special import expit
from datasets.dataset_downstream import ECGDatasetDownstream
from models.finetuner import FineTuner, get_backbone

from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight


def compute_hinge_loss(model, features, labels):
    """
    显式计算当前模型的 hinge loss。
    Args:
        model: 训练中的 SGDClassifier 模型。
        features: 输入特征矩阵。
        labels: 对应的标签数组。
    Returns:
        loss: 当前模型的 hinge loss。
    """
    # 获取模型权重和偏置
    weights = model.coef_  # 权重 w
    bias = model.intercept_  # 偏置 b

    # 计算模型的决策输出
    raw_decision = np.dot(features, weights.T) + bias  # w · x + b

    # 计算 hinge loss
    loss = np.maximum(0, 1 - labels[:, np.newaxis] * raw_decision)  # max(0, 1 - y (w · x + b))
    avg_loss = np.mean(loss)  # 平均损失

    return avg_loss


def logits_to_probability(logits, scale_factor=None):
    """
    对 logits 进行合理的缩放，避免数值极端导致评估问题。
    Args:
        logits (numpy array): SVM 的决策边界分数。
        scale_factor (float): 缩放因子（如果为 None，则自动计算为最大绝对值）。
    Returns:
        scaled_logits (numpy array): 缩放后的 logits。
    """
    if scale_factor is None:
        scale_factor = np.max(np.abs(logits))  # 使用 logits 的最大绝对值
    scaled_logits = logits / scale_factor * 10  # 将范围压缩到 [-10, 10]
    probability = 1 / (1 + np.exp(-scaled_logits))  # 应用 sigmoid

    return probability


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
                                        train_offline_features_file=os.path.join(args.data_root, args.train_offline_features_file) if args.train_offline_features_file is not None else None,
                                        valid_offline_features_file=os.path.join(args.data_root, args.valid_offline_features_file) if args.valid_offline_features_file is not None else None,
                                        test_offline_features_file=os.path.join(args.data_root, args.test_offline_features_file) if args.test_offline_features_file is not None else None,
                                        )

    # handle class imbalance
    sample_weights = train_ds.get_sample_weights()
    sampler = None

    if sample_weights is not None:
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

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

    # device, using cpu
    print_log('Using CPU', logger=logger)

    if "multi" not in args.downstream_task:
        task_info = {args.downstream_task: args.num_classes}
    else:
        multi_task_ids = train_ds.get_label_ids()
        task_info = {}
        for task_name in multi_task_ids:
            task_classes = 2
            task_info.update({task_name: task_classes})
    
    assert len(task_info.keys()) > 0, "multi-task setting not implemented!"

    svm_model = SGDClassifier(loss="hinge", learning_rate="optimal", tol=None)

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
                "f1_score": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "specificity": 0.0,
                "mcc": -1.0,
            }
            avg_best_metrics.update({
                "sensitivity": 0.0,
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

    _task_name = next(iter(task_info.keys()))
    save_path = os.path.join(args.experiment_path, 'best-checkpoint.pkl')

    if not args.test_only:
        print_log("==================== start training ====================", logger=logger)
        global_step = 0  # 初始化迭代数量

        metric_logger = MetricLogger(delimiter="  ")
        header = 'Training Iterations'
        while global_step < args.total_iterations:
            for it, batch in enumerate(metric_logger.log_every(train_dataloader, 100, header, logger)):
                
                if global_step >= args.total_iterations:
                    break
                
                features = batch['ecg_feature'].numpy()

                # Check if y is a dictionary (multi-task situation) and handle appropriately
                if isinstance(batch['label'], dict):
                    for key in batch['label']:
                        batch['label'][key] = batch['label'][key].squeeze()
                else:
                    batch['label'] = batch['label'].squeeze()

                y = batch['label']
                _y = y[_task_name]
                _y = np.where(_y == 0, -1, _y)  # 将标签 0 转为 -1

                # 计算类别权重
                class_weights = compute_class_weight('balanced', classes=np.unique(_y), y=_y)
                weights_dict = {label: weight for label, weight in zip(np.unique(_y), class_weights)}
                # 基于类别权重生成样本权重
                sample_weight = np.array([weights_dict[label] for label in _y])
                svm_model.partial_fit(features, _y, classes=np.unique(_y), sample_weight=sample_weight)

                task_losses = {
                    task: compute_hinge_loss(svm_model, features, _y)
                    for task in task_info.keys()
                }

                # 计算总损失，假设在多任务情况下，损失是每个任务损失的总和
                total_loss = sum(task_losses.values())

                logit = svm_model.decision_function(features)
                logit = logits_to_probability(logit)
                logits = {_task_name: logit}


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

                global_step += 1  # 增加迭代计数

                # 验证过程
                if global_step % args.val_iter == 0:
                    logits_dict = {task: [] for task in task_info.keys()}
                    labels_dict = {task: [] for task in task_info.keys()}
                    loss_dict = {task: [] for task in task_info.keys()}
                    # pid_list = []
                    with torch.no_grad():
                        val_metric_logger = MetricLogger(delimiter="  ")
                        for it, batch in enumerate(metric_logger.log_every(valid_dataloader, 100, header, logger)):

                            pid = batch['filename']
                            if isinstance(batch['label'], dict):
                                for key in batch['label']:
                                    batch['label'][key] = batch['label'][key]
                            else:
                                batch['label'] = batch['label']

                            y = batch['label']
                            features = batch['ecg_feature'].numpy()
                            # Check if y is a dictionary (multi-task situation) and handle appropriately
                            if isinstance(y, dict):
                                for key in y:
                                    y[key] = y[key].squeeze()
                            else:
                                y = y.squeeze()

                            _y = y[_task_name]
                            _y = np.where(_y == 0, -1, _y)  # 将标签 0 转为 -1

                            logit = svm_model.decision_function(features)  # 决策边界分数
                            logit = logits_to_probability(logit)
                            logits = {_task_name: logit}
                            task_losses = {
                                task: compute_hinge_loss(svm_model, features, _y)
                                for task in task_info.keys()
                            }

                            # Collect logits and labels
                            for task in task_info.keys():
                                task_logits = logits[task]
                                task_labels = y[task].float() if task == 'age' else y[task].long()  # Age task uses float labels

                                logits_dict[task].append(task_logits)
                                labels_dict[task].append(task_labels)
                                loss_dict[task].append(task_losses[task])

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
                            'model': svm_model,
                            'threshold': best_thresholds_dict,
                        }

                        if (args.downstream_task != 'age' and avg_metrics["pr_auc"] > avg_best_metrics["pr_auc"]) or \
                            (args.downstream_task == 'age' and avg_metrics["mean_squared_error"] < avg_best_metrics["mean_squared_error"]):
                                with open(save_path, 'wb') as f:
                                    save_dict = pickle.dump(save_dict, f)

                        # Update and log average best metrics if any metric's average improved
                        update_and_record_best_metrics(avg_metrics, avg_best_metrics, val_res_dict, best_metrics, writer, global_step)
                        print("============================")


    # test
    print_log("==================== start testing ====================", logger=logger)
    # load pre-trained img encoder

    with open(save_path, 'rb') as f:
        save_dict = pickle.load(f)
    svm_model = save_dict['model']
    best_thresholds_dict = save_dict['threshold']

    # # internal validation
    # hospital = 'internal'
    # test_model(
    #         test_ds=test_ds,
    #         svm_model=svm_model,
    #         task_info=task_info,
    #         hospital=hospital,
    #         logger=logger,
    #         writer=writer,
    #         args=args,
    #         threshold=best_thresholds_dict,
    #         save_flag=True,
    #         )

    # Save validation logits using the selected best checkpoint and validation-derived F1 threshold
    run_inference_and_save_logits(
        eval_ds=valid_ds,
        svm_model=svm_model,
        task_info=task_info,
        split_name='valid',
        logger=logger,
        writer=writer,
        args=args,
        threshold=best_thresholds_dict,
        save_flag=True,
    )

    # Save internal test logits using the same checkpoint and threshold
    run_inference_and_save_logits(
        eval_ds=test_ds,
        svm_model=svm_model,
        task_info=task_info,
        split_name='internal_test',
        logger=logger,
        writer=writer,
        args=args,
        threshold=best_thresholds_dict,
        save_flag=True,
    )


# def test_model(test_ds, svm_model, task_info, hospital, logger, writer, args, threshold, save_flag=False):
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

#     _task_name = next(iter(task_info.keys()))
#     _threshold = threshold[_task_name]

#     with torch.no_grad():
#         test_metric_logger = MetricLogger(delimiter="  ")
#         header = 'Test:'
        
#         for batch in test_metric_logger.log_every(test_dataloader, 10, header, logger):
#             pid = batch['filename']
#             # batch['ecg_data'] = batch['ecg_data']
#             features = batch['ecg_feature'].numpy()
            
#             if isinstance(batch['label'], dict):
#                 for key in batch['label']:
#                     batch['label'][key] = batch['label'][key]
#             else:
#                 batch['label'] = batch['label']
            
#             y = batch['label']
            
#             if isinstance(y, dict):
#                 for key in y:
#                     y[key] = y[key].squeeze()
#             else:
#                 y = y.squeeze()

#             logit = svm_model.decision_function(features)  # 决策边界分数
#             logit = logits_to_probability(logit)
#             logits = {_task_name: logit}
            
            
#             for task in task_info.keys(): # only one key
#                 task_logits = logits[task]
#                 task_labels = y[task].float() if task == 'age' else y[task].long()  # Age task uses float labels
#                 task_labels = task_labels.numpy()
                
#                 logits_dict[task].append(task_logits)
#                 labels_dict[task].append(task_labels)

#                 # 计算预测结果的正确与否
#                 # preds = np.argmax(task_logits, axis=1)
#                 preds = np.where(task_logits > _threshold, 1, 0).astype(np.int32)
#                 correct = preds == task_labels
#                 task_logits_list = task_logits.tolist()

#                 num_samples_in_batch = len(pid)
#                 task_name_list = [task] * num_samples_in_batch

#                 # print(preds, '\n', task_labels, '\n', correct, '\n', task_logits_list)
#                 result_list.extend(zip(pid, task_name_list, task_labels, preds, correct, task_logits_list))
            
        
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
#                 csv_writer = csv.writer(file)
#                 # csv_writer.writerow(['PID', 'Label', 'Pred Label', 'Correct', 'Logits'])
#                 csv_writer .writerow(['PID', 'Task', 'Label', 'Pred_Label', 'Correct', 'Logits'])
#                 csv_writer.writerows(result_list)


def run_inference_and_save_logits(eval_ds, svm_model, task_info, split_name, logger, writer, args, threshold, save_flag=False):
    eval_dataloader = DataLoader(
        eval_ds,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print_log(
        f"==================== {split_name} dataset load complete ====================: {len(eval_ds)} samples",
        logger=logger
    )

    eval_label_distribution = eval_ds.get_task_info()
    log_dict_to_logger(eval_label_distribution, pre=f"{split_name} data distribution", logger=logger)

    logits_dict = {task: [] for task in task_info.keys()}
    labels_dict = {task: [] for task in task_info.keys()}
    result_list = []

    _task_name = next(iter(task_info.keys()))
    _threshold = threshold[_task_name]

    with torch.no_grad():
        eval_metric_logger = MetricLogger(delimiter="  ")
        header = f'{split_name}:'

        for batch in eval_metric_logger.log_every(eval_dataloader, 10, header, logger):
            pid = batch['filename']
            features = batch['ecg_feature'].numpy()

            if isinstance(batch['label'], dict):
                for key in batch['label']:
                    batch['label'][key] = batch['label'][key]
            else:
                batch['label'] = batch['label']

            y = batch['label']

            if isinstance(y, dict):
                for key in y:
                    y[key] = y[key].squeeze()
            else:
                y = y.squeeze()

            logit = svm_model.decision_function(features)
            logit = logits_to_probability(logit)
            logits = {_task_name: logit}

            for task in task_info.keys():
                task_logits = logits[task]
                task_labels = y[task].float() if task == 'age' else y[task].long()
                task_labels = task_labels.numpy()

                logits_dict[task].append(task_logits)
                labels_dict[task].append(task_labels)

                preds = np.where(task_logits > _threshold, 1, 0).astype(np.int32)
                correct = preds == task_labels
                task_logits_list = task_logits.tolist()

                num_samples_in_batch = len(pid)
                task_name_list = [task] * num_samples_in_batch

                result_list.extend(zip(pid, task_name_list, task_labels, preds, correct, task_logits_list))

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
