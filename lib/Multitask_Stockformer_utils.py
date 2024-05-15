import logging
import numpy as np
import pandas as pd
import os
import pickle
import sys
import torch
import math
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import csv
# sys.path.append('/root/autodl-tmp/Stockformer/Stockformer_run/')
# from DILATE.loss.dilate_loss import dilate_loss
from torch.utils.data import Dataset

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(reg_pred, reg_label, class_pred, class_label):
    with np.errstate(divide='ignore', invalid='ignore'):
        # 回归任务的度量计算
        mask = np.not_equal(reg_label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(reg_pred, reg_label)).astype(np.float32)
        wape = np.divide(np.sum(mae), np.sum(reg_label))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, reg_label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        
        # 分类任务的准确率计算
        pred_classes = np.argmax(class_pred, axis=-1)
        correct = (pred_classes == class_label).astype(np.float32)
        acc = np.mean(correct)

    return acc, mae, rmse, mape


# 初始化交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()

def _compute_class_loss(y_true, y_predicted):
    # 展平 y_predicted 和 y_true
    y_predicted_flat = y_predicted.view(-1, y_predicted.size(-1))  # [batch_size * seq_len * num_nodes, num_classes]
    y_true_flat = y_true.view(-1).long()  # 转换为长整型

    # 计算损失
    loss = criterion(y_predicted_flat, y_true_flat)
    return loss


def _compute_regression_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def disentangle(data, w, j):
    # Disentangle
    dwt = DWT1DForward(wave=w, J=j)
    idwt = DWT1DInverse(wave=w)
    torch_traffic = torch.from_numpy(data).transpose(1,-1).reshape(data.shape[0]*data.shape[2], -1).unsqueeze(1)
    torch_trafficl, torch_traffich = dwt(torch_traffic.float())
    placeholderh = torch.zeros(torch_trafficl.shape)
    placeholderl = []
    for i in range(j):
        placeholderl.append(torch.zeros(torch_traffich[i].shape))
    torch_trafficl = idwt((torch_trafficl, placeholderl)).reshape(data.shape[0],data.shape[2],1,-1).squeeze(2).transpose(1,2)
    torch_traffich = idwt((placeholderh, torch_traffich)).reshape(data.shape[0],data.shape[2],1,-1).squeeze(2).transpose(1,2)
    trafficl = torch_trafficl.numpy()
    traffich = torch_traffich.numpy()
    return trafficl, traffich

def generate_temporal_embeddings(num_step, args):
    TE = np.zeros([num_step, 2])
    startd = (3 - 1) * 21
    df = 12
    startt = 0
    for i in range(num_step):
        TE[i, 0] = startd // 21
        TE[i, 1] = startt
        startd = (startd + 1) % (df * 21)
        startt = (startt + 1) % 21
    return TE

class StockDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        # Load data
        Traffic = np.load(args.traffic_file)['result']
        indicator = np.load(args.indicator_file)['result']
        # path = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/Stock_CN_2021-02-01_2023-12-29_Alpha_360/Alpha_360_2021-02-01_2023-12-29'
        path = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/Stock_CN_2021-06-04_2024-01-30/Alpha_360_2021-06-04_2024-01-30'
        files = os.listdir(path)
        data_list = []
        for file in files:
            file_path = os.path.join(path, file)
            df = pd.read_csv(file_path, index_col=0)
            arr = np.expand_dims(df.values, axis=2)
            data_list.append(arr)
        concatenated_arr = np.concatenate(data_list, axis=2)
        bonus_all = concatenated_arr
        num_step = Traffic.shape[0]
        train_steps = round(args.train_ratio * num_step)
        test_steps = round(args.test_ratio * num_step)
        val_steps = num_step - train_steps - test_steps
        TE = generate_temporal_embeddings(num_step, args)
        if mode == 'train':
            data_slice = slice(None, train_steps)
        elif mode == 'val':
            data_slice = slice(train_steps, train_steps + val_steps)
        else:  # mode == 'test'
            data_slice = slice(-test_steps, None)
        self.data = Traffic[data_slice]
        self.indicator = indicator[data_slice]
        self.bonus_all = bonus_all[data_slice]
        self.TE = TE[data_slice]
        self.X, self.Y = self.seq2instance(self.data, args.T1, args.T2)
        self.XL, self.XH = disentangle(self.X, args.w, args.j)
        self.YL, self.YH = disentangle(self.Y, args.w, args.j)
        self.indicator_X, self.indicator_Y = self.seq2instance(self.indicator, args.T1, args.T2)
        self.bonus_X, self.bonus_Y = self.bonus_seq2instance(self.bonus_all, args.T1, args.T2)
        self.TE = self.seq2instance(self.TE, args.T1, args.T2)
        self.TE = np.concatenate(self.TE, axis=1).astype(np.int32)
        # Adding the infea attribute based on bonus_all
        self.infea = bonus_all.shape[-1] + 2  # Last dimension of bonus_all plus one

    def __getitem__(self, index):
        return {
            'X': self.X[index],
            'X_low': self.XL[index],
            'X_high': self.XH[index],
            'indicator_X': self.indicator_X[index],
            'bonus_X': self.bonus_X[index],
            'TE': self.TE[index]
        }, {
            'Y': self.Y[index],
            'Y_low': self.YL[index],
            'Y_high': self.YH[index],
            'indicator_Y': self.indicator_Y[index],
            'bonus_Y': self.bonus_Y[index]
        }

    def __len__(self):
        return len(self.X)

    def seq2instance(self, data, P, Q):
        num_step, dims = data.shape
        num_sample = num_step - P - Q + 1
        x = np.zeros((num_sample, P, dims))
        y = np.zeros((num_sample, Q, dims))
        for i in range(num_sample):
            x[i] = data[i:i+P]
            y[i] = data[i+P:i+P+Q]
        return x, y

    def bonus_seq2instance(self, data, P, Q):
        num_step, dims, N = data.shape
        num_sample = num_step - P - Q + 1
        x = np.zeros((num_sample, P, dims, N))
        y = np.zeros((num_sample, Q, dims, N))
        for i in range(num_sample):
            x[i] = data[i:i+P]
            y[i] = data[i+P:i+P+Q]
        return x, y


def save_to_csv(file_path, data):
    # Check if the directory exists, if not, create it
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Write data to CSV
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)
    print(f"Data saved to {file_path}")  # You might want to replace this with your logging method