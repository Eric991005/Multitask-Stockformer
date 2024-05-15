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
from lib.Multitask_Stockformer_utils import log_string, _compute_regression_loss, _compute_class_loss, metric, save_to_csv, StockDataset
from lib.graph_utils import loadGraph
from Stockformermodel.Multitask_Stockformer_models import Stockformer

import os
from torch.utils.tensorboard import SummaryWriter

# 初始化解析器
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')

# 首次解析，仅获取config文件
args, unknown = parser.parse_known_args()  # 使用known_args来避免与后续添加的参数冲突

# 读取配置文件
config = configparser.ConfigParser()
config.read(args.config)

# 添加其它配置参数
parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
parser.add_argument('--seed', type=int, default=config['train']['seed'])
parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
parser.add_argument('--max_epoch', type=int, default=config['train']['max_epoch'])
parser.add_argument('--learning_rate', type=float, default=config['train']['learning_rate'])

parser.add_argument('--Dataset', default=config['data']['dataset'])
parser.add_argument('--T1', type=int, default=config['data']['T1'])
parser.add_argument('--T2', type=int, default=config['data']['T2'])
parser.add_argument('--train_ratio', type=float, default=config['data']['train_ratio'])
parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'])
parser.add_argument('--test_ratio', type=float, default=config['data']['test_ratio'])

parser.add_argument('--L', type=int, default=config['param']['layers'])
parser.add_argument('--h', type=int, default=config['param']['heads'])
parser.add_argument('--d', type=int, default=config['param']['dims'])
parser.add_argument('--j', type=int, default=config['param']['level'])
parser.add_argument('--s', type=float, default=config['param']['samples'])
parser.add_argument('--w', default=config['param']['wave'])

parser.add_argument('--traffic_file', default=config['file']['traffic'])
parser.add_argument('--indicator_file', default=config['file']['indicator'])
parser.add_argument('--adj_file', default=config['file']['adj'])
parser.add_argument('--adjgat_file', default=config['file']['adjgat'])
parser.add_argument('--model_file', default=config['file']['model'])
parser.add_argument('--log_file', default=config['file']['log'])

# 最终解析参数
args = parser.parse_args()

# 检查并创建日志文件目录
log_directory = os.path.dirname(args.log_file)
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    print(f"Directory created for log file: {log_directory}")
    
log = open(args.log_file, 'w')

# 检查并创建模型文件目录
model_directory = os.path.dirname(args.model_file)
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    print(f"Directory created for model file: {model_directory}")


# # 现在安全地打开日志文件写入
# with open(args.log_file, 'w') as log:
#     log.write("Logging has started.\n")
# print(f"Log file is ready to write at {args.log_file}")

# 确认模型文件路径准备就绪（这里仅确认路径，不创建文件）
print(f"Model file path is ready at {args.model_file}")


device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

tensorboard_folder = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/runs/Multitask_Stockformer/Stock_CN_2021-06-04_2024-01-30'

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

def res(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat, epoch, log, tensor_writer):
    model.eval()
    num_val = valXL.shape[0]
    num_batch = math.ceil(num_val / args.batch_size)

    pred_class = []
    pred_regress = []
    label_class = []
    label_regress = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(valXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(valXH[start_idx : end_idx]).float().to(device)
                xc = torch.from_numpy(valXC[start_idx : end_idx]).float().to(device)
                te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)
                bonus = torch.from_numpy(bonus_valX[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                yc = valYC[start_idx : end_idx]

                hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(xl, xh, te, bonus, xc, adjgat)

                pred_class.append(hat_y_class.cpu().numpy())
                pred_regress.append(hat_y_regress.cpu().numpy())
                label_class.append(yc)
                label_regress.append(y)
    
    pred_class = np.concatenate(pred_class, axis=0)
    pred_regress = np.concatenate(pred_regress, axis=0)
    label_class = np.concatenate(label_class, axis=0)
    label_regress = np.concatenate(label_regress, axis=0)

    accs = []
    maes = []
    rmses = []
    mapes = []

    # 假设第二维是时间维度
    for i in range(pred_class.shape[1]):
        acc, mae, rmse, mape = metric(pred_regress[:, i, :], label_regress[:, i, :], pred_class[:, i, :], label_class[:, i, :])
        accs.append(acc)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, f'step {i+1}, acc: {acc:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')

    avg_acc = np.mean(accs)
    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    avg_mape = np.mean(mapes)
    log_string(log, f'average, acc: {avg_acc:.4f}, mae: {avg_mae:.4f}, rmse: {avg_rmse:.4f}, mape: {avg_mape:.4f}')
    

    # Optional: Log to TensorBoard
    tensor_writer.add_scalar('Val/Average_Accuracy', avg_acc, epoch)
    tensor_writer.add_scalar('Val/Average_MAE', avg_mae, epoch)
    tensor_writer.add_scalar('Val/Average_RMSE', avg_rmse, epoch)
    tensor_writer.add_scalar('Val/Average_MAPE', avg_mape, epoch)
    
    return avg_acc, avg_mae, avg_rmse, avg_mape

def test_res(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat):
    model.eval()
    num_val = valXL.shape[0]
    num_batch = math.ceil(num_val / args.batch_size)

    pred_class = []
    pred_regress = []
    label_class = []
    label_regress = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(valXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(valXH[start_idx : end_idx]).float().to(device)
                xc = torch.from_numpy(valXC[start_idx : end_idx]).float().to(device)
                te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)
                bonus = torch.from_numpy(bonus_valX[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                yc = valYC[start_idx : end_idx]
                

                hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(xl, xh, te, bonus, xc, adjgat)

                pred_class.append(hat_y_class.cpu().numpy())
                pred_regress.append(hat_y_regress.cpu().numpy())  # 假设回归任务的输出需要反标准化
                label_class.append(yc)
                label_regress.append(y)
    
    pred_class = np.concatenate(pred_class, axis=0)
    pred_regress = np.concatenate(pred_regress, axis=0)
    label_class = np.concatenate(label_class, axis=0)
    label_regress = np.concatenate(label_regress, axis=0)

    accs = []
    maes = []
    rmses = []
    mapes = []

    for i in range(pred_regress.shape[1]):  # 假设第二维是时间维度
        acc, mae, rmse, mape = metric(pred_regress[:, i, :], label_regress[:, i, :], pred_class[:, i, :], label_class[:, i, :])
        accs.append(acc)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log,'step %d, acc: %.4f, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, acc, mae, rmse, mape))
    
    # 计算平均指标
    avg_acc = np.mean(accs)
    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    avg_mape = np.mean(mapes)
    log_string(log, 'average, acc: %.4f, mae: %.4f, rmse: %.4f, mape: %.4f' % (avg_acc, avg_mae, avg_rmse, avg_mape))
    
    # 保存分类任务的最后一个时间步的预测和标签
    save_to_csv('/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/Multitask_output_2021-06-04_2024-01-30/classification/classification_pred_last_step.csv', pred_class[:, -1, :])
    save_to_csv('/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/Multitask_output_2021-06-04_2024-01-30/classification/classification_label_last_step.csv', label_class[:, -1])

    # 保存回归任务的最后一个时间步的预测和标签
    save_to_csv('/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/Multitask_output_2021-06-04_2024-01-30/regression/regression_pred_last_step.csv', pred_regress[:, -1, :])
    save_to_csv('/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/output/Multitask_output_2021-06-04_2024-01-30/regression/regression_label_last_step.csv', label_regress[:, -1])

    return avg_acc, avg_mae, avg_rmse, avg_mape

def train(model, trainXL, trainXH, trainXC, bonus_trainX, trainTE, trainY, trainYL, trainYC, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat):
    num_train = trainXL.shape[0]
    # best_composite_score = 0.5
    best_mae = float('inf')
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
        trainXC = trainXC[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        trainYL = trainYL[permutation]
        trainYC = trainYC[permutation]
        bonus_trainX = bonus_trainX[permutation]
        num_batch = math.ceil(num_train / args.batch_size)

        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(trainXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(trainXH[start_idx : end_idx]).float().to(device)
                xc = torch.from_numpy(trainXC[start_idx : end_idx]).float().to(device)
                y = torch.from_numpy(trainY[start_idx : end_idx]).float().to(device)
                yl = torch.from_numpy(trainYL[start_idx : end_idx]).float().to(device)
                yc = torch.from_numpy(trainYC[start_idx : end_idx]).float().to(device)
                te = torch.from_numpy(trainTE[start_idx : end_idx]).to(device)
                bonus = torch.from_numpy(bonus_trainX[start_idx : end_idx]).float().to(device)
                
                
                optimizer.zero_grad()

                hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(xl, xh, te, bonus, xc, adjgat)

                loss_regress = _compute_regression_loss(y, hat_y_regress) + _compute_regression_loss(yl, hat_y_l_regress)
                loss_class = _compute_class_loss(yc, hat_y_class) + _compute_class_loss(yc, hat_y_l_class)
                
                # epsilon = 1e-8  # 防止除以零

                # # 计算权重（以损失的反比例为基础）
                # weight_regress = 1 / (loss_regress.item() + epsilon)
                # weight_class = 1 / (loss_class.item() + epsilon)

                # # 归一化权重，使得总和为1
                # weights_sum = weight_regress + weight_class
                # w1 = weight_regress / weights_sum
                # w2 = weight_class / weights_sum

                # 应用权重到损失函数
                # loss = w1*loss_regress + w2*loss_class
                loss = loss_regress + loss_class


                loss.backward()
                
                # # 梯度裁剪前打印梯度范围
                # max_grad = max(p.grad.data.abs().max() for p in model.parameters() if p.grad is not None)
                # print(f"Max grad before clipping: {max_grad}")
                
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                train_l_sum += loss.cpu().item()
                batch_count += 1
                pbar.update(1)

        log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
              % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))

        tensor_writer.add_scalar('training loss', train_l_sum / batch_count, epoch)

        # acc, mae, rmse, mape = res(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat, epoch, log, tensor_writer)
        # lr_scheduler.step(acc)  # 根据需要选择适当的指标进行调整
        # # 检查是否达到了新的最佳综合评分
        # # 检查是否达到了新的最佳准确率
        # if acc > best_composite_score:  # 现在我们希望“最高”准确率
        #     best_composite_score = acc  # 更新最佳综合评分
        #     # 保存具有最佳综合评分的模型
        #     torch.save(model.state_dict(), args.model_file)
        #     log_string(log, f'Epoch {epoch}: New best accuracy: {best_composite_score:.4f}, Model saved.')
        
        
        # 假设你的 res 函数返回准确率（acc），MAE，RMSE 和 MAPE
        acc, mae, rmse, mape = res(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat, epoch, log, tensor_writer)
        
        # 使用 MAE 作为学习率调度器的度量
        lr_scheduler.step(mae)  # 传递 mae 而不是 acc

        # 检查是否得到了更低的 MAE，这意味着模型的表现更好了
        if mae < best_mae:  # 寻找最小 MAE
            best_mae = mae  # 更新最佳 MAE 记录
            # 保存具有最佳 mae 的模型状态
            torch.save(model.state_dict(), args.model_file)
            log_string(log, f'Epoch {epoch}: New best mae: {best_mae:.4f}, Model saved.')


def test(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat):
    try:
        model.load_state_dict(torch.load(args.model_file))
        total_params = sum(p.numel() for p in model.parameters())
        log_string(log, 'Total parameters: {}'.format(total_params))
    except EOFError:
        print(f"Error: Unable to load model state dictionary from file {args.model_file}. File may be empty or corrupted.")
        return

    acc, mae, rmse, mape = test_res(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat)
    return acc, mae, rmse, mape


if __name__ == '__main__':
    log_string(log, "loading data....")
    outfea_class = 2
    outfea_regress = 1
    # trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, testXL, testXH, testTE, testY, bonus_all_trainX, bonus_all_valX, bonus_all_testX, infeature = loadData(args)
    train_dataset = StockDataset(args, mode='train')
    val_dataset = StockDataset(args, mode='val')
    test_dataset = StockDataset(args, mode='test')
    # get data
    # train data
    trainXL = train_dataset.XL
    trainXH = train_dataset.XH
    trainXC = train_dataset.indicator_X
    trainTE = train_dataset.TE
    trainY = train_dataset.Y
    trainYL = train_dataset.YL
    trainYC = train_dataset.indicator_Y
    bonus_trainX = train_dataset.bonus_X
    # val data
    valXL = val_dataset.XL
    valXH = val_dataset.XH
    valXC = val_dataset.indicator_X
    valTE = val_dataset.TE
    valY = val_dataset.Y
    valYL = val_dataset.YL
    valYC = val_dataset.indicator_Y
    bonus_valX = val_dataset.bonus_X
    # test data
    testXL = test_dataset.XL
    testXH = test_dataset.XH
    testXC = test_dataset.indicator_X
    testTE = test_dataset.TE
    testY = test_dataset.Y
    testYL = test_dataset.YL
    testYC = test_dataset.indicator_Y
    bonus_testX = test_dataset.bonus_X
    # infeature number
    infeature = train_dataset.infea
    # adj, graphwave = loadGraph(args)
    adjgat = loadGraph(args)
    adjgat = torch.from_numpy(adjgat).float().to(device)
    log_string(log, "loading end....")

    log_string(log, "constructing model begin....")
    model = Stockformer(infeature, args.h*args.d, outfea_class, outfea_regress, args.L, args.h, args.d, args.s, args.T1, args.T2, device).to(device)
    log_string(log, "constructing model end....")

    log_string(log, "training begin....")
    train(model, trainXL, trainXH, trainXC, bonus_trainX, trainTE, trainY, trainYL, trainYC, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat)
    log_string(log, "training end....")

    log_string(log, "testing begin....")
    test(model, testXL, testXH, testXC, bonus_testX, testTE, testY, testYC, adjgat)
    log_string(log, "testing end....")
