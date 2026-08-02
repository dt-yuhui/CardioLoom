import pandas as pd
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, h5_file, label_df, info_df, indices, num_leads):
        """
        初始化 ECGDataset
        
        Parameters:
        h5_file (str): h5 数据文件路径
        label_df (DataFrame): 预先读取的标签 DataFrame
        info_df (DataFrame): 预先读取的信息 DataFrame
        indices (list): 用于划分训练集或验证集的索引列表
        """
        self.h5_file = h5_file
        self.indices = indices


        # 将文件名映射到报告
        info_filename_to_report = dict(zip(info_df['Filename'], info_df['english_report']))
        label_df['report'] = label_df['Filename'].map(info_filename_to_report)
        
        self.labels_df = label_df
        self.h5_data = h5py.File(h5_file, 'r')['ECGs']

        if num_leads not in [8, 12]:
            raise ValueError("num_leads must be either 8 or 12.")
        self.num_leads = num_leads

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        row = self.labels_df.loc[data_idx]
        
        ecg_data = self.h5_data[data_idx]
        ecg_data = self.convert_ecg_data(ecg_data=ecg_data)
        filename = row['Filename']
        report = row['report']

        return {
            'ecg_data': torch.tensor(ecg_data).float(),
            'filename': filename,
            'report': report
        }
    
    def convert_ecg_data(self, ecg_data):
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
        
        return ecg_data
    
    @staticmethod
    def create_datasets(h5_file, label_file, info_file, train_index_file, valid_index_file, num_leads):
        """
        创建训练和验证数据集
        
        Parameters:
        h5_file (str): h5 数据文件路径
        label_file (str): CSV 标签文件路径
        info_file (str): 包含 report 的 CSV 文件路径
        train_index_file (str): 训练集索引文件路径
        valid_index_file (str): 验证集索引文件路径
        
        Returns:
        (train_dataset, valid_dataset): Tuple of train and validation datasets
        """
        # 读取标签文件和信息文件
        label_df = pd.read_csv(label_file, low_memory=False)
        info_df = pd.read_csv(info_file, low_memory=False)
        
        # 读取索引文件
        train_indices = pd.read_csv(train_index_file)['index'].tolist()
        valid_indices = pd.read_csv(valid_index_file)['index'].tolist()
        
        # 创建训练和验证数据集
        train_dataset = ECGDataset(h5_file, label_df, info_df, train_indices, num_leads)
        valid_dataset = ECGDataset(h5_file, label_df, info_df, valid_indices, num_leads)

        print(f'train size: {len(train_dataset)}')
        print(f'val size: {len(valid_dataset)}')
        
        return train_dataset, valid_dataset

