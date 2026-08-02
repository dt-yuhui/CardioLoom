import os
import math
import struct
import h5py

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from .ecg import bandpass_filtering, notch_filtering, resampling, zero_padding
from scipy.signal import resample

VALVE_MULTI_LABEL_TASKS = ['AS', 'AR', 'TR', 'MR', 'MS']

def ad2mv(b):
    value = struct.unpack("h", b)
    return value[0]


def convert(ecg_path):
    ecg_data = []
    try:
        with open(ecg_path, "rb") as infile:
            # 获取文件大小
            file_size = os.path.getsize(ecg_path)
            # 计算移动后的位置
            new_position = file_size - 480000
            # 移动文件指针到指定位置

            for i in range(0, 8):
                infile.seek(new_position + 50 * 2 * 500 + i * 60000)
                lead_data = []
                for pos in range(0, 500 * 10):
                    b = infile.read(2)
                    value = ad2mv(b)
                    lead_data.append(value)

                lead_data_array = np.array(lead_data)
                ecg_data.append(lead_data_array)

        ecg_data = np.vstack(ecg_data) * 0.00244
        ecg_data = np.transpose(ecg_data)

        ecg_data = bandpass_filtering(ecg_data, 500, 0.5, 100)
        ecg_data = notch_filtering(ecg_data, 500, 50)

        ecg_data = np.transpose(ecg_data)
    except Exception as e:
        print(ecg_path, e)

    return ecg_data


class ShiyuanDataset(Dataset):
    def __init__(self, 
                 task, 
                 num_leads=8,
                 downsample=False,
                 data_dir='data/external/shiyuan',
                 label_csv='data/external/shiyuan/labels.csv',
                 exclude_label_ids=None,
                 offline_features_file=None
                 ):
        super().__init__()
        self.df = pd.read_csv(label_csv, low_memory=False)
        self.indices = self.df.index
        self.data_dir = data_dir
        self.task = task

        if num_leads not in [8, 12]:
            raise ValueError("num_leads must be either 8 or 12.")
        self.num_leads = num_leads
        self.downsample = downsample

        self.exclude_label_ids = exclude_label_ids if exclude_label_ids is not None else []
        self.exclude_columns = [f"echo_label_{i}" for i in self.exclude_label_ids]

        # 加载离线特征并构建索引映射
        if offline_features_file is not None:
            self.offline_features_file = h5py.File(offline_features_file, 'r')
            self.offline_features = self.offline_features_file['features']
            h5_indices = self.offline_features_file['indices'][:]  # 读取离线文件中的主索引
            self.offline_index_lookup = {idx: pos for pos, idx in enumerate(h5_indices)}
        else:
            self.offline_features = None

        # 根据task统一处理标签选择
        # [task_1, task_2, ...]
        self.label_columns = self.get_task_label_columns(task, self.df.columns, self.exclude_columns)

        # 生成文件名到其任务标签的映射
        # {filename: label}
        info_filename_to_task_label = self.get_task_labels(self.df, self.label_columns, task)
        # print(info_filename_to_task_label)

        # 合并操作以加速处理
        # multi-task
        if 'multi_label' in task:
            for col in self.label_columns:
                self.df[col] = self.df['ECGPath'].map(info_filename_to_task_label.get).apply(lambda x: x.get(col, -1) if x else -1)
                # 填充 NaN 为 -1
                self.df[col] = self.df[col].replace(np.nan, -1)
        # single-task
        else:
            self.df['label'] = self.df['ECGPath'].map(info_filename_to_task_label).fillna(-1)

        # 生成有效标签的索引
        valid_indices = self.get_valid_indices(self.df, task)
        indices = self.df.index.tolist()
        self.indices = list(set(indices).intersection(valid_indices))

    def get_task_label_columns(self, task, all_columns, exclude_columns):
        if task == 'lvef_label':
            return ["lvef_label"]
        elif "echo_label" in task:
            return [task]
        elif task in ["MS", "MR", "AS", "AR", "TS", "TR", "PS", "PR"]:
            return [task]
        elif task == 'echo_multi_label':
            return [col for col in all_columns if "echo_label" in col and col not in exclude_columns]
        elif task == 'valve_multi_label':
            return [col for col in VALVE_MULTI_LABEL_TASKS]
        else:
            raise ValueError(f"Unknown task: {task}")
        
    def get_task_labels(self, info_df, label_columns, task):
        if 'multi_label' not in task:
            return dict(zip(info_df['ECGPath'], info_df[label_columns[0]]))
        else:
            # 对于多标签任务，返回字典形式的DataFrame
            return info_df.set_index('ECGPath')[label_columns].to_dict('index')
        
    def get_label_ids(self, ):

        return self.label_columns
    
    def get_task_info(self):
        if "multi_label" in self.task:
            all_labels = self.df.loc[self.indices, self.label_columns]
            label_distribution = {col: all_labels[col].value_counts().to_dict() for col in self.label_columns}
        else:
            label_distribution = self.df.loc[self.indices, 'label'].value_counts().to_dict()

        return label_distribution
    
    def get_valid_indices(self, label_df, task):
        if "multi_label" in task:
            # 多标签任务: 找出所有标签列中没有 -1 的行索引
            valid_mask = (label_df[self.label_columns] != -1).all(axis=1)
            return label_df.index[valid_mask]
        else:
            # 单标签任务: 过滤掉标签为 -1 的样本
            return label_df.index[label_df['label'] != -1]

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        row = self.df.iloc[data_idx]
        ecg_path = os.path.join(self.data_dir, row['ECGPath'])
        if self.offline_features is not None:
            feature_pos = self.offline_index_lookup.get(data_idx, None)
            ecg_feature = self.offline_features[feature_pos] if feature_pos is not None else None
        else:
            ecg_feature = None

        try:
            ecg_data = convert(ecg_path)  # 假设 convert 是你定义的用来加载 ECG 信号的函数
            ecg_data = torch.from_numpy(ecg_data.copy()).float()
            error_occurred = False
        except Exception as e:
            print(f"Error loading file {ecg_path}: {e}")
            ecg_data = torch.zeros((8, 5000))  # 返回一个空的张量作为占位符
            error_occurred = True

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

        if "multi_label" in self.task:
            labels = row[self.label_columns].values
            labels = np.array(labels).astype(float)
            labels = torch.tensor(labels).long()
        else:
            label = row['label']
            labels = torch.tensor([label])

        return {
            'ecg_data': ecg_data,
            'filename': ecg_path,
            'data_idx':  data_idx,
            'label': {task_id: label.item() for task_id, label in zip(self.label_columns, labels)},
            'error_occurred': error_occurred,
            'ecg_feature': torch.tensor(ecg_feature).float() if ecg_feature is not None else [],
        }

    def __len__(self):
        return len(self.indices)
