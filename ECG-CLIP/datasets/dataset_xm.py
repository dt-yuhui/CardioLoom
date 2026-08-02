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


def read_dat_and_preprocess(file_path, gain=0.00244):
    f = open(file_path, "rb")
    all_ecg = []
    for i in range(8):
        lead_data = []
        for pos in range(500 * 10):
            b = f.read(2)
            if not b: break
            value = struct.unpack("h", b)[0]
            lead_data.append(value)
        all_ecg.append(np.array(lead_data))
    f.close()
    
    # 1. 将列表堆叠成矩阵，并应用增益转换为物理单位 (mV)
    ecg_data = np.vstack(all_ecg) * gain
    # 2. 转置矩阵，使形状变为 (样本数, 导联数)
    ecg_data = np.transpose(ecg_data)
    ecg_data = bandpass_filtering(ecg_data, 500, 0.5, 100)
    ecg_data = notch_filtering(ecg_data, 500, 50)
    ecg_data = np.transpose(ecg_data)
    
    return ecg_data


class XiamenDataset(Dataset):
    def __init__(self, 
                 task,
                 num_leads=8,
                 downsample=False,
                 data_dir='data/external/xiamen',
                 label_csv='data/external/xiamen/labels.csv',
                 exclude_label_ids=None):
        super().__init__()
        self.df = pd.read_csv(label_csv, low_memory=False)
        self.indices = self.df.index
        self.data_dir = data_dir
        self.task = task
        self.exclude_label_ids = exclude_label_ids if exclude_label_ids is not None else []
        self.exclude_columns = [f"ecg_label_{i}" for i in self.exclude_label_ids]


        if num_leads not in [8, 12]:
            raise ValueError("num_leads must be either 8 or 12.")
        self.num_leads = num_leads
        self.downsample = downsample

        # 根据task统一处理标签选择
        # [task_1, task_2, ...]
        self.label_columns = self.get_task_label_columns(task, self.df.columns, self.exclude_columns)
        print(self.label_columns)

        # 生成文件名到其任务标签的映射
        # {filename: label}
        info_filename_to_task_label = self.get_task_labels(self.df, self.label_columns, task)
        # print(info_filename_to_task_label)

        # 合并操作以加速处理
        if 'multi_label' not in task:
            self.df['label'] = self.df['Filename'].map(info_filename_to_task_label).fillna(-1)
    
    def get_task_label_columns(self, task, all_columns, exclude_columns):
        if "ecg_label" in task:
            return [task]
        elif task == 'ecg_multi_label':
            return [col for col in all_columns if "ecg_label" in col and col not in exclude_columns]
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def get_task_labels(self, info_df, label_columns, task):
        if 'multi_label' not in task:
            return dict(zip(info_df['Filename'], info_df[label_columns[0]]))
        
    def get_label_ids(self, ):

        return self.label_columns
    
    def get_task_info(self):
        if "multi_label" in self.task:
            all_labels = self.df.loc[self.indices, self.label_columns]
            label_distribution = {col: all_labels[col].value_counts().to_dict() for col in self.label_columns}
        else:
            label_distribution = self.df.loc[self.indices, 'label'].value_counts().to_dict()

        return label_distribution

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_path = os.path.join(self.data_dir, row['Filename']+'.dat')

        try:
            ecg_data = read_dat_and_preprocess(ecg_path) 
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
            'label': {task_id: label.item() for task_id, label in zip(self.label_columns, labels)},
            'error_occurred': error_occurred
        }

    def __len__(self):
        return len(self.df)
