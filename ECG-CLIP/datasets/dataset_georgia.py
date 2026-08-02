import os

import numpy as np
import pandas as pd
import xmltodict as xtd
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

from .ecg import bandpass_filtering, notch_filtering, resampling, zero_padding
from scipy.signal import resample
import wfdb


class GeorgiaDataset(Dataset):
    def __init__(self, 
                 task,
                 num_leads=8,
                 downsample=False,
                 data_dir='data/external/georgia',
                 label_csv='data/external/georgia/labels.csv',
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

        self.lead_order = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        if self.num_leads == 8:
            self.desired_leads = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
        else:
            self.desired_leads = self.lead_order
        self.lead_indices = [self.lead_order.index(lead) for lead in self.desired_leads]

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
            self.df['label'] = self.df['ecg_data_path'].map(info_filename_to_task_label).fillna(-1)


    def get_task_label_columns(self, task, all_columns, exclude_columns):
        if "ecg_label" in task:
            return [task]
        elif task == 'ecg_multi_label':
            return [col for col in all_columns if "ecg_label" in col and col not in exclude_columns]
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def get_task_labels(self, info_df, label_columns, task):
        if 'multi_label' not in task:
            return dict(zip(info_df['ecg_data_path'], info_df[label_columns[0]]))
        
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
        ecg_path = os.path.join(self.data_dir, row['ecg_data_path'])

        try:
            ecg_data, meta_data = wfdb.rdsamp(ecg_path)
            # some ecg signal in this dataset is of shape (2500, 12)
            ecg_data = zero_padding(ecg_signal=ecg_data, dst_len=5000)
            ecg_data = ecg_data.T
            ecg_data = torch.from_numpy(ecg_data.copy()).float()
            error_occurred = False
        except Exception as e:
            print(f"Error loading file {ecg_path}: {e}")
            ecg_data = torch.zeros((8, 5000))  # 返回一个空的张量作为占位符
            error_occurred = True

        ecg_data = ecg_data[self.lead_indices, :]
        # print(ecg_data)
        has_nan = np.isnan(ecg_data).any()
        if has_nan:
            print(ecg_data, ecg_path)

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
