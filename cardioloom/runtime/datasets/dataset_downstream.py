import numpy as np
import json
import os
from torch.utils.data import Dataset
import pandas as pd
import pickle
from tqdm import tqdm
import torch
import re
from sklearn.model_selection import train_test_split
import argparse
import h5py
import random
import time
from scipy.signal import resample


VALVE_MULTI_LABEL_TASKS = ['AS', 'AR', 'TR', 'MR', 'MS']


class ECGDatasetDownstream(Dataset):
    def __init__(self, h5_file, label_df, info_df, indices, task, ecg_or_echo='ecg', num_leads=8, downsample=False, exclude_label_ids=None, offline_features_file=None):
        """
        初始化 ECGDataset
        
        Parameters:
        h5_file (str): h5 数据文件路径
        label_df (DataFrame): 预先读取的标签 DataFrame
        info_df (DataFrame): 预先读取的信息 DataFrame
        indices (list): 用于划分训练集或验证集的索引列表
        """
        self.h5_file = h5_file
        # self.indices = indices
        self.task = task
        self.h5_data = h5py.File(h5_file, 'r')['ECGs']
        self.exclude_label_ids = exclude_label_ids if exclude_label_ids is not None else []
        self.ecg_or_echo = ecg_or_echo

        if num_leads not in [8, 12]:
            raise ValueError("num_leads must be either 8 or 12.")
        self.num_leads = num_leads
        self.downsample = downsample

        # 加载离线特征并构建索引映射
        if offline_features_file is not None:
            self.offline_features_file = h5py.File(offline_features_file, 'r')
            self.offline_features = self.offline_features_file['features']
            h5_indices = self.offline_features_file['indices'][:]  # 读取离线文件中的主索引
            self.offline_index_lookup = {idx: pos for pos, idx in enumerate(h5_indices)}
        else:
            self.offline_features = None

        if self.ecg_or_echo == 'ecg':
            self.exclude_columns = [f"ecg_label_{i}" for i in self.exclude_label_ids]
        else:
            self.exclude_columns = [f"echo_label_{i}" for i in self.exclude_label_ids]

        # 根据task统一处理标签选择
        # [task_1, task_2, ...]
        self.label_columns = self.get_task_label_columns(task, info_df.columns, self.exclude_columns)
        print(self.label_columns)

        # 生成文件名到其任务标签的映射
        # {filename: label}
        info_filename_to_task_label = self.get_task_labels(info_df, self.label_columns, task)
        # print(info_filename_to_task_label)

        # 合并操作以加速处理
        # multi-task
        if 'multi_label' in task:
            for col in self.label_columns:
                label_df[col] = label_df['Filename'].map(info_filename_to_task_label.get).apply(lambda x: x.get(col, -1) if x else -1)
                # 填充 NaN 为 -1
                label_df[col] = label_df[col].replace(np.nan, -1)
        # single-task
        else:
            label_df['label'] = label_df['Filename'].map(info_filename_to_task_label).fillna(-1)

        self.labels_df = label_df

        # 生成有效标签的索引
        valid_indices = self.get_valid_indices(label_df, task)
        self.indices = list(set(indices).intersection(valid_indices))

        # 计算样本权重（仅限于非年龄和多标签任务）
        if task not in ['age', 'ecg_multi_label', 'echo_multi_label', 'valve_multi_label']:
            self.setup_class_weights_and_sample_weights(label_df)
        else:
            self.class_weights = self.sample_weights = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        row = self.labels_df.loc[data_idx]
        ecg_data = self.h5_data[data_idx]

        # G-1: Check if 12-lead conversion is needed
        if self.num_leads == 12:
            # G-2: Ensure input is 8-lead before trying to convert
            if ecg_data.shape[0] != 8:
                raise RuntimeError(f"12-lead conversion requires 8-lead input, but got shape {ecg_data.shape}")
            
            # G-3: Calculate the 4 additional leads
            lead_I = ecg_data[0, :]
            lead_II = ecg_data[1, :]
            
            lead_III = lead_II - lead_I
            lead_aVR = -(lead_I + lead_II) / 2
            lead_aVL = lead_I - (lead_II / 2)
            lead_aVF = lead_II - (lead_I / 2)
            
            # G-4: Stack all 12 leads in the standard order
            # Standard order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
            v_leads = ecg_data[2:, :] # These are V1-V6
            
            # Use np.stack and then reshape, or vstack
            calculated_leads = np.stack([lead_III, lead_aVR, lead_aVL, lead_aVF], axis=0)
            
            # Reassemble the ecg_data variable with all 12 leads
            ecg_data = np.vstack([
                lead_I[np.newaxis, :], 
                lead_II[np.newaxis, :], 
                calculated_leads, 
                v_leads
            ])

        if self.downsample:
            original_length = ecg_data.shape[1]
            new_length = int(original_length / 5)
            ecg_data = resample(ecg_data, new_length, axis=1)

        if self.offline_features is not None:
            feature_pos = self.offline_index_lookup.get(data_idx, None)
            ecg_feature = self.offline_features[feature_pos] if feature_pos is not None else None
        else:
            ecg_feature = None

        filename = row['Filename']
        
        if "multi_label" in self.task:
            labels = row[self.label_columns].values
            labels = np.array(labels).astype(float)
            labels = torch.tensor(labels).long()
        else:
            label = row['label']
            if self.task == 'age':
                label = max(0, min(100, label))
                label = label / 100.0
            labels = [torch.tensor(label)]

        return {
            'ecg_data': torch.tensor(ecg_data).float(),
            'filename': filename,
            'data_idx':  data_idx,
            'label': {task_id: label.item() for task_id, label in zip(self.label_columns, labels)},
            'ecg_feature': torch.tensor(ecg_feature).float() if ecg_feature is not None else [],
        }
    
    def setup_class_weights_and_sample_weights(self, label_df):
        filtered_label_df = label_df.loc[self.indices].reset_index(drop=True)
        self.class_counts = filtered_label_df['label'].value_counts().sort_index()
        self.class_weights = 1.0 / torch.tensor(self.class_counts.values).float()
        self.sample_weights = filtered_label_df['label'].map(dict(zip(self.class_counts.index, self.class_weights)))
    
    def get_sample_weights(self, ):
        
        return self.sample_weights

    def get_class_weights(self, ):

        return self.class_weights
    
    def get_label_ids(self, ):

        return self.label_columns
    
    def get_task_info(self):
        if "multi_label" in self.task:
            all_labels = self.labels_df.loc[self.indices, self.label_columns]
            label_distribution = {col: all_labels[col].value_counts().to_dict() for col in self.label_columns}
        else:
            label_distribution = self.labels_df.loc[self.indices, 'label'].value_counts().to_dict()
        return label_distribution

    def get_task_label_columns(self, task, all_columns, exclude_columns):
        if task == "gender":
            return ["Gender(M-0/F-1)"]
        elif task == 'lvef_label':
            return ["lvef_label"]
        elif task in ["MS", "MR", "AS", "AR", "TS", "TR", "PS", "PR"]:
            return [task]
        elif task == 'af_label':
            return ["af_label"]
        elif task == 'age':
            return ["Age"]
        elif "ecg_label" in task:
            return [task]
        elif "echo_label" in task:
            return [task]
        elif task == 'ecg_multi_label':
            return [col for col in all_columns if "ecg_label" in col and col not in exclude_columns]
        elif task == 'echo_multi_label':
            return [col for col in all_columns if "echo_label" in col and col not in exclude_columns]
        elif task == 'valve_multi_label':
            return [col for col in VALVE_MULTI_LABEL_TASKS]
        else:
            raise ValueError(f"Unknown task: {task}")

    def get_valid_indices(self, label_df, task):
        if "multi_label" in task:
            # 多标签任务: 找出所有标签列中没有 -1 的行索引
            valid_mask = (label_df[self.label_columns] != -1).all(axis=1)
            return label_df.index[valid_mask]
        else:
            # 单标签任务: 过滤掉标签为 -1 的样本
            return label_df.index[label_df['label'] != -1]

    def get_task_labels(self, info_df, label_columns, task):
        if "multi_label" in task:
            # 对于多标签任务，返回字典形式的DataFrame
            return info_df.set_index('Filename')[label_columns].to_dict('index')
        else:  
            return dict(zip(info_df['Filename'], info_df[label_columns[0]]))

    @staticmethod
    def create_datasets(h5_file, 
                        label_file, 
                        info_file, 
                        train_index_file, 
                        valid_index_file, 
                        test_index_file, 
                        task, 
                        ecg_or_echo='ecg',
                        num_leads=8,
                        downsample=False,
                        exclude_label_ids=None, 
                        random_split_seed=None, 
                        sample_ratio=1.0, 
                        train_offline_features_file=None,
                        valid_offline_features_file=None,
                        test_offline_features_file=None):
        """
        创建训练和验证数据集
        
        Parameters:
        h5_file (str): h5 数据文件路径
        label_file (str): CSV 标签文件路径
        info_file (str): 包含 label 的 CSV 文件路径
        train_index_file (str): 训练集索引文件路径
        valid_index_file (str): 验证集索引文件路径
        
        Returns:
        (train_dataset, valid_dataset, test_dataset): Tuple of train, validation and test datasets
        """

        def _sample_filenames(filenames, ratio):

            sample_size = int(len(filenames) * ratio)
            return random.sample(filenames, sample_size)
        
        # 读取标签文件和信息文件
        label_df = pd.read_csv(label_file, low_memory=False)
        info_df = pd.read_csv(info_file, low_memory=False)
        
        # 读取索引文件
        train_indices = pd.read_csv(train_index_file)['index'].tolist()
        valid_indices = pd.read_csv(valid_index_file)['index'].tolist()
        test_indices = pd.read_csv(test_index_file)['index'].tolist()

        if random_split_seed is not None:
            # 合并 train 和 valid 索引
            combined_indices = train_indices + valid_indices
            
            # 按照 7:2 的比例重新划分 train 和 valid
            train_indices, valid_indices = train_test_split(combined_indices, test_size=2/9, random_state=random_split_seed)
        
        # if sample_ratio < 1.0:
        #     train_indices = _sample_filenames(train_indices, sample_ratio)

        # 创建训练和验证数据集
        train_dataset = ECGDatasetDownstream(h5_file, label_df, info_df, train_indices, task, ecg_or_echo, num_leads, downsample, exclude_label_ids, train_offline_features_file)
        valid_dataset = ECGDatasetDownstream(h5_file, label_df, info_df, valid_indices, task, ecg_or_echo, num_leads, downsample, exclude_label_ids, valid_offline_features_file)
        test_dataset = ECGDatasetDownstream(h5_file, label_df, info_df, test_indices, task, ecg_or_echo, num_leads, downsample, exclude_label_ids, test_offline_features_file)

        print(f'train size: {len(train_dataset)}')
        print(f'val size: {len(valid_dataset)}')
        print(f'test size: {len(test_dataset)}')
        
        return train_dataset, valid_dataset, test_dataset

