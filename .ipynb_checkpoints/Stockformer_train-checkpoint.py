from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import configparser
import math
import csv
import random
from pytorch_wavelets import DWT1DForward, DWT1DInverse


from Stockformer_code.lib import CN_utils
from Stockformer_code.lib.CN_utils import log_string, loadData, _compute_loss, _compute_dilate_loss, metric, save_to_csv
from lib.graph_utils import loadGraph
from Stockformermodel.Stockformermodels import Stockformer

import os
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)

parser.add_argument('--cuda', type=str, 
            default=config['train']['cuda'])
parser.add_argument('--seed', type = int, 
            default = config['train']['seed'])
parser.add_argument('--batch_size', type = int, 
            default = config['train']['batch_size'])
parser.add_argument('--max_epoch', type = int, 
            default = config['train']['max_epoch'])
parser.add_argument('--learning_rate', type=float, 
            default = config['train']['learning_rate'])

parser.add_argument('--Dataset', default = config['data']['dataset'])
parser.add_argument('--T1', type = int, 
            default = config['data']['T1'])
parser.add_argument('--T2', type = int, 
            default = config['data']['T2'])
parser.add_argument('--train_ratio', type = float, 
            default = config['data']['train_ratio'])
parser.add_argument('--val_ratio', type = float, 
            default = config['data']['val_ratio'])
parser.add_argument('--test_ratio', type = float, 
            default = config['data']['test_ratio'])

parser.add_argument('--L', type = int,
            default = config['param']['layers'])
parser.add_argument('--h', type = int,
            default = config['param']['heads'])
parser.add_argument('--d', type = int, 
            default = config['param']['dims'])
parser.add_argument('--j', type = int, 
            default = config['param']['level'])
parser.add_argument('--s', type = float,
            default = config['param']['samples'])
parser.add_argument('--w',
            default = config['param']['wave'])

parser.add_argument('--traffic_file', default = config['file']['traffic'])
parser.add_argument('--adj_file', default = config['file']['adj'])
parser.add_argument('--adjgat_file', default = config['file']['adjgat'])
parser.add_argument('--model_file', default = config['file']['model'])
parser.add_argument('--log_file', default = config['file']['log'])

args = parser.parse_args()

log = open(args.log_file, 'w')

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

tensorboard_folder = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/runs/CN_Data'

# Check and create the main TensorBoard folder
if not os.path.exists(tensorboard_folder):
    os.makedirs(tensorboard_folder)
    log_string(log, f"Folder created: {tensorboard_folder}")
else:
    log_string(log, f"Folder already exists: {tensorboard_folder}")

# Determine the name for the new subfolder
subfolders = [f.name for f in os.scandir(tensorboard_folder) if f.is_dir()]
versions = [int(folder.replace('version', '')) for folder in subfolders if folder.startswith('version')]
next_version = 0 if not versions else max(versions) + 1
new_folder = os.path.join(tensorboard_folder, f'version{next_version}')

# Create the new subfolder
if not os.path.exists(new_folder):
    os.makedirs(new_folder)
    log_string(log, f"Subfolder created: {new_folder}")

# Create a SummaryWriter instance pointing to the new subfolder
tensor_writer = SummaryWriter(new_folder)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

def res(model, valXL, valXH, valTE, valY, bonus_all_valX, adjgat, epoch, tensor_writer):
    model.eval()
    num_val = valXL.shape[0]
    num_batch = math.ceil(num_val / args.batch_size)

    pred = []
    label = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(valXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(valXH[start_idx : end_idx]).float().to(device)
                bonus = torch.from_numpy(bonus_all_valX[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)

                y_hat, y_hat_l = model(xl, xh, te, bonus, adjgat)

                pred.append(y_hat.cpu().numpy())
                label.append(y)
    
    pred = np.concatenate(pred, axis = 0)
    label = np.concatenate(label, axis = 0)

    maes = []
    rmses = []
    mapes = []
    accuracys = []

    for i in range(pred.shape[1]):
        mae, rmse , mape, accuracy = metric(pred[:,i,:], label[:,i,:])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        accuracys.append(accuracy)
        log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f, accuracy: %.4f' % (i+1, mae, rmse, mape, accuracy))
    
    mae, rmse, mape, accuracy = metric(pred, label)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    accuracys.append(accuracy)
    log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f, accuracy: %.4f' % (mae, rmse, mape, accuracy))
    
    # 记录 MAE, RMSE, MAPE
    for i in range(len(maes)):
        tensor_writer.add_scalar(f'Val/MAE_Step{i+1}', maes[i], epoch)
        tensor_writer.add_scalar(f'Val/RMSE_Step{i+1}', rmses[i], epoch)
        tensor_writer.add_scalar(f'Val/MAPE_Step{i+1}', mapes[i], epoch)

    tensor_writer.add_scalar('Val/Average_MAE', maes[-1], epoch)
    tensor_writer.add_scalar('Val/Average_RMSE', rmses[-1], epoch)
    tensor_writer.add_scalar('Val/Average_MAPE', mapes[-1], epoch)
    
    return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0), np.stack(accuracys, 0)

def test_res(model, valXL, valXH, valTE, valY, bonus_all_valX, adjgat):
    model.eval()
    num_val = valXL.shape[0]
    num_batch = math.ceil(num_val / args.batch_size)

    pred = []
    label = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(valXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(valXH[start_idx : end_idx]).float().to(device)
                bonus = torch.from_numpy(bonus_all_valX[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)

                y_hat, y_hat_l = model(xl, xh, te, bonus, adjgat)

                pred.append(y_hat.cpu().numpy())
                label.append(y)
    
    pred = np.concatenate(pred, axis = 0)
    label = np.concatenate(label, axis = 0)

    maes = []
    rmses = []
    mapes = []
    accuracys = []

    for i in range(pred.shape[1]):
        mae, rmse , mape, accuracy = metric(pred[:,i,:], label[:,i,:])
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        accuracys.append(accuracy)
        log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f, accuracy: %.4f' % (i+1, mae, rmse, mape, accuracy))
    
    mae, rmse, mape, accuracy = metric(pred, label)
    maes.append(mae)
    rmses.append(rmse)
    mapes.append(mape)
    accuracys.append(accuracy)
    log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f, accuracy: %.4f' % (mae, rmse, mape, accuracy))
    
    # Save predictions for the first and second elements, and labels likewise
    save_to_csv('/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/CN_pred_embed.csv', pred[:, 0, :])
    # save_to_csv('/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/usa_second_pred.csv', pred[:, 1, :])
    save_to_csv('/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/CN_label_embed.csv', label[:, 0, :])
    # save_to_csv('/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/usa_second_label.csv', label[:, 1, :])


    return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0), np.stack(accuracys, 0)

def train(model, trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, bonus_all_trainX, bonus_all_valX, adjgat):
    num_train = trainXL.shape[0]
    min_loss = 0.5
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,    
                                    verbose=False, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=2e-6, eps=1e-08)
    
    for epoch in tqdm(range(1,args.max_epoch+1)):
        model.train()
        train_l_sum, batch_count, start = 0.0, 0, time.time()
        permutation = np.random.permutation(num_train)
        trainXL = trainXL[permutation]
        trainXH = trainXH[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        trainYL = trainYL[permutation]
        bonus_all_trainX = bonus_all_trainX[permutation]
        num_batch = math.ceil(num_train / args.batch_size)

        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(trainXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(trainXH[start_idx : end_idx]).float().to(device)
                y = torch.from_numpy(trainY[start_idx : end_idx]).float().to(device)
                yl = torch.from_numpy(trainYL[start_idx : end_idx]).float().to(device)
                te = torch.from_numpy(trainTE[start_idx : end_idx]).to(device)
                bonus = torch.from_numpy(bonus_all_trainX[start_idx : end_idx]).float().to(device)
                
                
                optimizer.zero_grad()

                y_hat, y_hat_l = model(xl, xh, te, bonus,adjgat)

                loss_mae = _compute_loss(y, y_hat) + _compute_loss(yl, y_hat_l)
                loss_dilate = _compute_dilate_loss(y, y_hat, alpha=0.5, gamma=0.001, device='cuda')
                # 计算权重（以损失的反比例为基础）
                weight_mae = 1 / loss_mae.item()
                weight_dilate = 1 / loss_dilate.item()

                # 归一化权重，使得总和为1
                weights_sum = weight_mae + weight_dilate
                w1 = weight_mae / weights_sum
                w2 = weight_dilate / weights_sum
                loss = w1 * loss_mae + w2 * loss_dilate

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                
                train_l_sum += loss.cpu().item()
                batch_count += 1
                pbar.update(1)

        log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
              % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))

        tensor_writer.add_scalar('training loss', train_l_sum / batch_count, epoch)

        mae, rmse, mape, accuracy = res(model, valXL, valXH, valTE, valY, bonus_all_valX, adjgat, epoch, tensor_writer)
        lr_scheduler.step(abs(mape[-1]))
        if abs(mape[-1]) < min_loss:
            min_loss = abs(mape[-1])
            torch.save(model.state_dict(), args.model_file)


def test(model, valXL, valXH, valTE, valY, bonus_all_testX, adjgat):
    try:
        model.load_state_dict(torch.load(args.model_file))
    except EOFError:
        print(f"Error: Unable to load model state dictionary from file {args.model_file}. File may be empty or corrupted.")
        return

    mae, rmse, mape, accuracy = test_res(model, valXL, valXH, valTE, valY, bonus_all_testX, adjgat)
    return mae, rmse, mape, accuracy


if __name__ == '__main__':
    log_string(log, "loading data....")
    trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, testXL, testXH, testTE, testY, bonus_all_trainX, bonus_all_valX, bonus_all_testX, infeature = loadData(args)
    # adj, graphwave = loadGraph(args)
    adjgat = loadGraph(args)
    adjgat = torch.from_numpy(adjgat).float().to(device)
    log_string(log, "loading end....")

    log_string(log, "constructing model begin....")
    
    model = Stockformer(infeature, args.h*args.d, args.L, args.h, args.d, args.s, args.T1, args.T2, device).to(device)
    log_string(log, "constructing model end....")

    log_string(log, "training begin....")
    train(model, trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, bonus_all_trainX, bonus_all_valX, adjgat)
    log_string(log, "training end....")

    log_string(log, "testing begin....")
    test(model, testXL, testXH, testTE, testY, bonus_all_testX, adjgat)
    log_string(log, "testing end....")
