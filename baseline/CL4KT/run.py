import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from load_data import getLoader
from config import CL4KTConfig


def run_epoch(is_train, path, model, optimizer, batch_size, criterion, seq_len, mask_prob, crop_prob, permute_prob,
              replace_prob, negative_prob):
    total_loss = []
    total_correct = 0
    total_num = 0
    labels = []
    outputs = []

    if is_train:
        model.train()
    else:
        model.eval()

    data_loader = getLoader(is_train, path, batch_size, seq_len, mask_prob, crop_prob, permute_prob, replace_prob,
                            negative_prob)

    # 会把数据集分为多个batch_size大小的数据（i表示数据的个数）
    # 调用load_data中的getitem方法取data
    for i, batch in tqdm(enumerate(data_loader), desc='加载中...', ncols=100):

        if is_train:
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            out_dict = model(batch)

            cl_loss = torch.mean(out_dict["cl_loss"])

            # 计算损失
            next_predict = torch.masked_select(out_dict["pred"], out_dict["mask"])
            next_true = torch.masked_select(out_dict["true"], out_dict["mask"])
            kt_loss = criterion(next_predict, next_true.float()) + CL4KTConfig.REG_CL * cl_loss

            # 反向传播
            kt_loss.backward()

            if CL4KTConfig.MAX_GRAD_NORM > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CL4KTConfig.MAX_GRAD_NORM)

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

                out_dict = model(batch)

                next_predict = torch.masked_select(out_dict["pred"], out_dict["mask"])
                next_true = torch.masked_select(out_dict["true"], out_dict["mask"])
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
