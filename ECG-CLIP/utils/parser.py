import os
import argparse
# import utils.util as util
from . import util
import numpy as np


def get_clip_args():
    parser = argparse.ArgumentParser()

    # Distributed training parameters
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--train_batch_size_per_gpu', type=int, default=768)
    parser.add_argument('--valid_batch_size_per_gpu', type=int, default=768)

    # 3D dataset parameters (with augmentation)
    parser.add_argument('--data_root', default="data", type=str, help='root directory containing the preprocessed dataset')
    parser.add_argument('--h5_file_path', default="ecg_data_all.h5", type=str, help='path of h5 data')
    parser.add_argument('--label_file_path', default="ecg_labels_all.csv", type=str, help='path of label')
    parser.add_argument('--info_file_path', default="all_information.csv", type=str, help='path of info')
    parser.add_argument('--train_indices_file_path', default="new_split_v2/pretrain_train_indices.csv", type=str, help='path of train indices')
    parser.add_argument('--valid_indices_file_path', default="new_split_v2/pretrain_valid_indices.csv", type=str, help='path of valid indices')
    parser.add_argument('--txt_transform', default=True, type=util.bool_flag, help='whether to shuffle sentence')

    # Model parameters 
    parser.add_argument('--text_model', default='ncbi/MedCPT-Query-Encoder', type=str,
                        help='Hugging Face model ID or local directory for the text encoder')
    parser.add_argument('--text_model_local_files_only', default=False, type=util.bool_flag,
                        help='load the text encoder only from local files/cache (useful on offline servers)')
    parser.add_argument('--ecg_model', default='resnet18', help='type of ecg encoder')
    parser.add_argument('--mlp_hidden_size', default=256, type=int, help='hidden channel of mlp')
    parser.add_argument('--projection_size', default=256, type=int, help='output channel of mlp')
    parser.add_argument('--num_leads', default=8, type=int, help='num of ecg leads')
    parser.add_argument('--freeze_text_encoder', default=False, type=util.bool_flag, help='whether to freeze text encoder')
    parser.add_argument('--text_encoder_freeze_layers', default=9, type=int, help='num of frozen layers in text encoder')

    # Loss paramters
    parser.add_argument('--uma_weight', default=1.0, type=float, help='weight of uma loss')

    # Training/Optimization parameters
    parser.add_argument('--resume', default=None, type=str, help='Path for continue training')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs of training.')
    parser.add_argument('--verbose_epoch', default=-1, type=int, help='Number of epochs to print verbose validation info.')

    parser.add_argument('--use_fp16', type=util.bool_flag, default=False, help="Whether or not to use half precision for training.")
    parser.add_argument("--lr", default=2e-4, type=float, help="Learning rate for train from scratch at the end of linear warmup (highest LR used during training).")
    parser.add_argument("--momentum", default=0.9, type=float, help='momentum for optimizer')
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="Initial value of the weight decay.")
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')

    # Output parameters
    parser.add_argument('--exp_name', default="ecg_clip_r18", type=str, help='experiment name used for logs and checkpoints')
    parser.add_argument('--log_name', default="ECG_CLIP", type=str, help='log name')
    parser.add_argument('--output_root', default="outputs/pretrain", type=str,
                        help='root directory for pretraining logs and checkpoints')
    args = parser.parse_args()

    args.experiment_path = os.path.join(args.output_root, args.exp_name)
    args.tfboard_path = os.path.join(args.output_root, 'tfboard', args.exp_name)

    create_experiment_dir(args)

    return args


def get_downstream_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--split_random_state', default=42, type=int, help='Random data split seed.')
    parser.add_argument('--exp_name', default="", type=str, help='Path to save logs, tensorboard and checkpoints.')
    parser.add_argument('--log_name', default="ECGCLIP", type=str, help='log name')
    parser.add_argument('--resume', default=None, type=str, help='Path for checkpoint')
    parser.add_argument('--in_features', default=512, type=int, help='Input features')
    parser.add_argument('--num_classes', default=2, type=int, help='num classes')
    parser.add_argument('--finetune_type', default="train_from_scratch", choices=["finetune", "linear_prob", "train_from_scratch", "pretrain", 'svm'], help='type of finetune')
    parser.add_argument('--cuda', type=util.bool_flag, default=True, help='enables CUDA training')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--downstream_task', default='gender', type=str,help="type of downstream task")
    parser.add_argument('--test_only', default=False, type=util.bool_flag, help="whether to tarin & valid")
    parser.add_argument('--exclude_label_ids', default=(5, 10, 12, 20, 22, 23, 25, 26, 30, 35, 37, 50, 54, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74), type=util.parse_tuple, help="name of hospitals of external validation")
    parser.add_argument('--rare_disease_label_ids', default=(5, 12, 20, 26, 30, 37, 50, 63, 65, 66, 71, 73, 74), type=util.parse_tuple, help="name of hospitals of external validation")
    parser.add_argument('--ecg_or_echo', default='ecg', type=str, help="name of hospitals of external validation")
    # dataset
    # 3D dataset parameters (with augmentation)
    parser.add_argument('--data_root', default="data", type=str, help='root directory containing the preprocessed dataset')
    parser.add_argument('--h5_file_path', default="ecg_data_all.h5", type=str, help='path of h5 data')
    parser.add_argument('--label_file_path', default="ecg_labels_all.csv", type=str, help='path of label')
    parser.add_argument('--info_file_path', default="all_information.csv", type=str, help='path of info')
    
    parser.add_argument('--train_indices_file_path', default="new_split_v2/downstream_train_indices.csv", type=str, help='path of train indices')
    parser.add_argument('--valid_indices_file_path', default="new_split_v2/downstream_valid_indices.csv", type=str, help='path of valid indices')
    parser.add_argument('--test_indices_file_path', default="new_split_v2/downstream_test_indices.csv", type=str, help='path of test indices')

    parser.add_argument('--train_offline_features_file', default=None, type=str, help='path of offline-saved train samples')
    parser.add_argument('--valid_offline_features_file', default=None, type=str, help='path of offline-saved valid samples')
    parser.add_argument('--test_offline_features_file', default=None, type=str, help='path of offline-saved test samples')

    parser.add_argument('--sample_ratio', default=1.0, type=float, help='sample ratio of training set')
    parser.add_argument('--random_split_seed', default=None, type=int, help='whether random split the pre-split dataset')

    parser.add_argument('--external_dataset', default='', type=str, help='name of external dataset')
    parser.add_argument('--num_leads', default=8, type=int, help='# of ecg leads')
    parser.add_argument('--downsample', default=False, type=util.bool_flag, help='whether to downsample frequency, if true 100 Hz otherwise 500 Hz')

    # model
    parser.add_argument('--ecg_model', default='resnet50', type=str, help="model type")
    parser.add_argument('--custom_pretrained_root', default="outputs/pretrain", type=str,
                        help="root directory containing ECG-CLIP pretraining experiments")
    parser.add_argument('--pretrained_exp_name', default=None, type=str, help="clip name")
    parser.add_argument('--pretrained_metric', default=None, type=str, choices=['loss', 'acc'], help="load which clip, best loss/acc")
    parser.add_argument('--pretrained_checkpoint', default=None, type=str,
                        help="explicit ECG-CLIP checkpoint path; overrides pretrained_exp_name/pretrained_metric")
    parser.add_argument('--merl_checkpoint', default=None, type=str,
                        help="path to a MERL checkpoint when --ecg_model merl")
    parser.add_argument('--ckpt_name', default='', type=str, help="ckpt name")

    parser.add_argument('--drop_path', default=0.0, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate at the end of linear warmup (highest LR used during training).")
    parser.add_argument('--weight_decay', type=float, default=5e-2, help="""Initial value of the weight decay.""")
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--val_iter', default=200, type=int)
    parser.add_argument('--total_iterations', default=4000, type=int)
    parser.add_argument('--output_root', default="outputs/downstream", type=str,
                        help='root directory for downstream logs and checkpoints')

    args = parser.parse_args()

    args.experiment_path = os.path.join(args.output_root, args.finetune_type, args.exp_name)
    args.tfboard_path = os.path.join(args.output_root, args.finetune_type, 'tfboard', args.exp_name)
    args.ckpt_path = os.path.join(args.output_root, args.finetune_type, args.ckpt_name)

    return args


def create_experiment_dir(args):
    if util.is_main_process():
        if not os.path.exists(args.experiment_path):
            os.makedirs(args.experiment_path, exist_ok=True)
            print('Create experiment path successfully at %s' % args.experiment_path)
        if not os.path.exists(args.tfboard_path):
            os.makedirs(args.tfboard_path, exist_ok=True)
            print('Create TFBoard path successfully at %s' % args.tfboard_path)
