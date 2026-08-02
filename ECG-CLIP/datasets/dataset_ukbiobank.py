import os

import numpy as np
import pandas as pd
import xmltodict as xtd
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

from .ecg import bandpass_filtering, notch_filtering, resampling, zero_padding
from scipy.signal import resample

def import_ecg(fdir, fname, save_image_to, lead_order):
    # f = open(fdir+fname, "rt")
    f = open(os.path.join(fdir, fname), "rt")
    raw_input = f.read()
    f.close()
    md = {'filename':fname}
    # extract ECG and metadata
    ecg, median_ecg, md = parse_xml(raw_input, lead_order, md)

    return ecg, median_ecg, md


def parse_xml(input_data, lead_order,md):
    """ Takes input as raw xml data read from file (Cardiosoft specification), returns list of 12 lead waveforms
    and metadata. """
    data = xtd.parse(input_data)['CardiologyXML']

    full_lead_nodes = [['StripData', 'WaveformData'], ['Strip', 'StripData', 'WaveformData']]
    median_lead_nodes = [['RestingECGMeasurements', 'MedianSamples', 'WaveformData'], []]
    full_leads = get_lead_data(data, full_lead_nodes, lead_order)
    median_leads = get_lead_data(data, median_lead_nodes, lead_order)

    md = get_metadata(data, md)
    # md['lead order'] = lead_order

    return np.array(full_leads, dtype=np.int16), np.array(median_leads, dtype=np.int16), md


def get_lead_data(data, nodes, lead_order):
    leads = [[] for i in range(12)]
    raw_lead_data = get_xml_node(data, nodes)

    # check number of leads
    if len(raw_lead_data) != 12:
        print('Warning: only {} leads found'.format(len(raw_lead_data)))

    # split data into 12 leads and convert string into individual values
    else:
        for i in range(len(raw_lead_data)):
            lead_n = lead_order.index(raw_lead_data[i]['@lead'].upper())
            lead_data_string = raw_lead_data[i]['#text']
            lead_vals = [int(x) for x in lead_data_string.split(",")]
            leads[lead_n] = lead_vals

    return leads


def get_xml_node(data, node_list):
    output = data
    try:
        for x in node_list[0]:
            output = output[x]
    except KeyError:
        output = data[:]
        try:
            for x in node_list[1]:
                output = output[x]
        except:
            raise ValueError('No lead data found!')
    return output


def get_metadata(data, md):
    patient_info = data['PatientInfo']
    md['PID'] = patient_info['PID']
    md['Age'] = patient_info['Age']['#text']
    BirthDateTime = patient_info['BirthDateTime']
    md['BirthDateTime'] = BirthDateTime['Year'] +'-'+BirthDateTime['Month'] +'-'+BirthDateTime['Day']
    md['Gender'] = patient_info['Gender']
    md['Race'] = patient_info['Race']
    md['Height'] = patient_info['Height']['#text']
    md['Weight'] = patient_info['Weight']['#text']
    md['PaceMaker'] = patient_info['PaceMaker']


    md['sample rate'] = float(data['StripData']['SampleRate']['#text'])
    # md['t scale'] = 1. / md['sample rate']
    # md['v scale'] = float(data['StripData']['Resolution']['#text'])
    # md['filter 50Hz'] = data['FilterSetting']['Filter50Hz']
    # md['filter 60Hz'] = data['FilterSetting']['Filter60Hz']
    # md['low pass'] = float(data['FilterSetting']['LowPass']['#text'])
    # md['high pass'] = float(data['FilterSetting']['HighPass']['#text'])

    # meas = data['RestingECGMeasurements']
    # md['Heart rate'] = meas['VentricularRate']['#text'] + ' ' + meas['VentricularRate']['@units']
    # md['P duration'] = meas['PDuration']['#text'] + ' ' + meas['PDuration']['@units']
    # md['PR interval'] = meas['PQInterval']['#text'] + ' ' + meas['PQInterval']['@units']
    # md['QRS duration'] = meas['QRSDuration']['#text'] + ' ' + meas['QRSDuration']['@units']
    # md['QT interval'] = meas['QTInterval']['#text'] + ' ' + meas['QTInterval']['@units']
    # md['QTc interval'] = meas['QTCInterval']['#text'] + ' ' + meas['QTCInterval']['@units']
    # md['P axis'] = meas['PAxis']['#text'] + ' ' + meas['PAxis']['@units']
    # md['R axis'] = meas['RAxis']['#text'] + ' ' + meas['RAxis']['@units']
    # md['T axis'] = meas['TAxis']['#text'] + ' ' + meas['TAxis']['@units']


    diagnosis = data['Interpretation']['Diagnosis']
    md['Diagnosis'] = diagnosis['DiagnosisText']
    return md


def main():
    file_dir = 'data/external/ukbiobank'
    df_out = pd.DataFrame()
    for fname in tqdm(os.listdir(file_dir)):
        # names and order of leads to import
        lead_order = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        # pathname of output for image file
        save_image_to = '1000022_20205_2_0.svg'
        try:
            # run example
            ecg, median_ecg, md = import_ecg(file_dir ,fname, save_image_to, lead_order)
            df_out = pd.concat([df_out,pd.DataFrame([md])])
        except Exception as e:
            print(fname, e)
    df_out.to_csv('ukbiobank.csv',index=None)


class UKBioBankDataset(Dataset):
    def __init__(self, 
                 task,
                 num_leads=8,
                 downsample=False,
                 data_dir='data/external/ukbiobank',
                 label_csv='data/external/ukbiobank/labels.csv',
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
            self.df['label'] = self.df['filename'].map(info_filename_to_task_label).fillna(-1)

    def convert_ecg(self, ecg_path):
        ecg, median_ecg, md = import_ecg(self.data_dir, ecg_path, None, self.lead_order)
        ecg_data = ecg[self.lead_indices, :]
        # 信号处理
        ecg_data = np.transpose(ecg_data) * 0.00244
        ecg_data = bandpass_filtering(ecg_data, 500, 0.5, 100)
        ecg_data = notch_filtering(ecg_data, 500, 50)
        # 返回处理后的信号
        ecg_data = np.transpose(ecg_data)

        return ecg_data
    
    def get_task_label_columns(self, task, all_columns, exclude_columns):
        if "ecg_label" in task:
            return [task]
        elif task == 'ecg_multi_label':
            return [col for col in all_columns if "ecg_label" in col and col not in exclude_columns]
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def get_task_labels(self, info_df, label_columns, task):
        if 'multi_label' not in task:
            return dict(zip(info_df['filename'], info_df[label_columns[0]]))
        
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
        ecg_path = os.path.join(self.data_dir, row['filename'])

        try:
            ecg_data = self.convert_ecg(ecg_path)  # 假设 convert 是你定义的用来加载 ECG 信号的函数
            ecg_data = torch.from_numpy(ecg_data.copy()).float()
            error_occurred = False
        except Exception as e:
            print(f"Error loading file {ecg_path}: {e}")
            ecg_data = torch.zeros((8, 5000))  # 返回一个空的张量作为占位符
            error_occurred = True

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



if __name__ == '__main__':
    main()
