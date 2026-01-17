import torch
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from torch.nn.functional import binary_cross_entropy
from load_data import getLoader


def run_epoch(is_train, path, model, optimizer, batch_size, min_problem_num, max_problem_num):
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
    for i, data in tqdm(enumerate(data_loader), desc='加载中...', ncols=100):
        use_problem, use_skill, use_ans, res_mask = data

        if is_train:
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            predict = model(use_skill[:, :-1], use_ans[:, :-1], use_skill[:, 1:])

            # 计算损失
            # predict、next_ans、mask都是80,199，mask与next_ans一一对应，都是去掉了第一位，并且值为true代表该位置是用户真正的交互数据，而值为false代表该位置是为了统一长度而填充的数据
            # 经过mask操作之后，把用户真正交互数据挑选出来了，而将为了统一长度填充的数据筛选出去了
            next_predict = torch.masked_select(predict, res_mask[:, 1:])
            next_true = torch.masked_select(use_ans[:, 1:], res_mask[:, 1:])
            kt_loss = binary_cross_entropy(next_predict, next_true.float())

            # 反向传播
            kt_loss.backward()

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

                predict = model(use_skill[:, :-1], use_ans[:, :-1], use_skill[:, 1:])

                next_predict = torch.masked_select(predict, res_mask[:, 1:])
                next_true = torch.masked_select(use_ans[:, 1:], res_mask[:, 1:])
                kt_loss = binary_cross_entropy(next_predict, next_true.float())

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
