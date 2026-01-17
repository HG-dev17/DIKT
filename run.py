import torch
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import torch.nn as nn
from load_data import getLoader


def run_epoch(is_train, path, model, optimizer, batch_size, min_problem_num,
              max_problem_num, grad_clip, criterion):
    total_loss = []
    total_correct = 0
    total_num = 0
    labels = []
    outputs = []

    if is_train:
        model.train()
    else:
        model.eval()

    data_loader = getLoader(is_train, path, batch_size, min_problem_num, max_problem_num)

    # 会把数据集分为多个batch_size大小的数据（i表示数据的个数）
    # 调用load_data中的getitem方法取data
    desc_text = '训练中' if is_train else '验证中'
    # 获取数据加载器的总长度用于显示进度
    total_batches = len(data_loader)
    for i, data in tqdm(enumerate(data_loader), desc=desc_text, total=total_batches, ncols=120, leave=True, position=2):
        use_problem, use_ans, res_mask, use_time = data

        if is_train:
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            predict = model(use_problem, use_ans, use_time)

            # 计算损失
            # 预测是对下一个位置的预测，所以需要shift
            # predict[i] 预测的是 use_ans[i+1]，所以需要对齐
            # 使用res_mask来过滤有效位置，但预测和真实值需要正确对齐
            # 预测位置i对应的是答案位置i（因为模型在位置i使用历史信息预测位置i的答案）
            next_predict = torch.masked_select(predict, res_mask)
            next_true = torch.masked_select(use_ans, res_mask)
            kt_loss = criterion(next_predict, next_true.float())

            kt_loss.backward()

            # 梯度剪切
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            # 更新参数
            optimizer.step()

            # total_loss存放着该批次的损失值，目的是为了对所有批次的损失值求平均
            total_loss.append(kt_loss.item())

            # 为了计算整体的acc，将该批次的长度以及该批次正确数添加到total_num和total_correct
            total_num += len(next_true)
            to_pred = (next_predict >= 0.5).long()
            total_correct += (next_true == to_pred).sum()

            # 为了计算整体的auc,将该批次的预测结果和真实标签添加到outputs和labels中
            labels.extend(next_true.view(-1).data.cpu().numpy())
            outputs.extend(next_predict.view(-1).data.cpu().numpy())
        else:
            with torch.no_grad():
                predict = model(use_problem, use_ans, use_time)

                # 预测位置i对应的是答案位置i（因为模型在位置i使用历史信息预测位置i的答案）
                next_predict = torch.masked_select(predict, res_mask)
                next_true = torch.masked_select(use_ans, res_mask)
                kt_loss = criterion(next_predict, next_true.float())

                total_loss.append(kt_loss.item())

                total_num += len(next_true)
                to_pred = (next_predict >= 0.5).long()
                total_correct += (next_true == to_pred).sum()

                labels.extend(next_true.view(-1).data.cpu().numpy())
                outputs.extend(next_predict.view(-1).data.cpu().numpy())

    loss = np.mean(total_loss)
    acc = total_correct.item() / total_num
    auc = metrics.roc_auc_score(labels, outputs)
    rmse = np.sqrt(metrics.mean_squared_error(y_true=labels, y_pred=outputs))
    return loss, acc, auc, rmse