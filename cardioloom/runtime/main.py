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

from utils.util import *

from utils.logger import *

from utils.loss import clip_loss

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
