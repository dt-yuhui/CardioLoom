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

from datasets.dataset_shiyuan import ShiyuanDataset

from sklearn.linear_model import SGDClassifier

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
    if args.external_dataset == 'shiyuan':
        # dataset & dataloader
        test_ds = ShiyuanDataset(
            task=args.downstream_task,
            exclude_label_ids=args.exclude_label_ids,
            offline_features_file=os.path.join(args.data_root, args.test_offline_features_file) if args.test_offline_features_file is not None else None,
            )

    # device, using cpu
    print_log('Using CPU', logger=logger)

    if "multi" not in args.downstream_task:
        task_info = {args.downstream_task: args.num_classes}
    else:
        multi_task_ids = test_ds.get_label_ids()
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
    save_path = os.path.join(args.ckpt_path, 'best-checkpoint.pkl')

    # test
    print_log("==================== start testing ====================", logger=logger)
    # load pre-trained img encoder

    with open(save_path, 'rb') as f:
        save_dict = pickle.load(f)
    svm_model = save_dict['model']
    best_thresholds_dict = save_dict['threshold']

    if args.external_dataset == 'shiyuan':
        task_to_remove = ['echo_label_24']
    external_dataset_task_info = {k: v for k, v in task_info.items() if k not in task_to_remove}

    # validation
    hospital = args.external_dataset
    test_model(
            test_ds=test_ds,
            svm_model=svm_model,
            task_info=external_dataset_task_info,
            hospital=hospital,
            logger=logger,
            writer=writer,
            args=args,
            threshold=best_thresholds_dict,
            save_flag=True,
            )


def test_model(test_ds, svm_model, task_info, hospital, logger, writer, args, threshold, save_flag=False):
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

    _task_name = next(iter(task_info.keys()))
    _threshold = threshold[_task_name]

    with torch.no_grad():
        test_metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'
        
        for batch in test_metric_logger.log_every(test_dataloader, 10, header, logger):
            pid = batch['filename']
            # batch['ecg_data'] = batch['ecg_data']
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

            logit = svm_model.decision_function(features)  # 决策边界分数
            logit = logits_to_probability(logit)
            logits = {_task_name: logit}
            
            for task in task_info.keys(): # only one key
                task_logits = logits[task]
                task_labels = y[task].float() if task == 'age' else y[task].long()  # Age task uses float labels
                task_labels = task_labels.numpy()
                
                logits_dict[task].append(task_logits)
                labels_dict[task].append(task_labels)

                # 计算预测结果的正确与否
                # preds = np.argmax(task_logits, axis=1)
                preds = np.where(task_logits > _threshold, 1, 0).astype(np.int32)
                correct = preds == task_labels
                task_logits_list = task_logits.tolist()

                num_samples_in_batch = len(pid)
                task_name_list = [task] * num_samples_in_batch

                # print(preds, '\n', task_labels, '\n', correct, '\n', task_logits_list)
                result_list.extend(zip(pid, task_name_list, task_labels, preds, correct, task_logits_list))
            
        
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
                csv_writer = csv.writer(file)
                # csv_writer.writerow(['PID', 'Label', 'Pred Label', 'Correct', 'Logits'])
                csv_writer .writerow(['PID', 'Task', 'Label', 'Pred_Label', 'Correct', 'Logits'])
                csv_writer.writerows(result_list)


if __name__ == '__main__':
    args = get_downstream_args()
    main(args)
