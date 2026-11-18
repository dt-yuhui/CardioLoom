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

import pandas as pd

from datasets.dataset_downstream import ECGDatasetDownstream

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
