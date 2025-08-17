import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable, grad
from load_data import getLoader

from config import BASICConfig, ATKTConfig


def run_epoch(is_train, path, model, optimizer, batch_size, min_problem_num,
              max_problem_num):
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
        use_skill, use_ans, res_mask = data

        if is_train:
            optimizer.zero_grad()

            predict, features = model(use_skill, use_ans)
            next_predict = (torch.masked_select(predict, res_mask[:,1:])).float()
            next_true = (torch.masked_select(use_ans[:,1:], res_mask[:,1:])).float()
            pred_loss = nn.BCELoss()(next_predict, next_true)

            features_grad = grad(pred_loss, features, retain_graph=True)
            p_adv = torch.FloatTensor(ATKTConfig.EPSILON * _l2_normalize_adv(features_grad[0].data))
            p_adv = Variable(p_adv).to(BASICConfig.DEVICE)

            predict_adv, features_adv = model(use_skill, use_ans, p_adv)
            next_predict_adv = (torch.masked_select(predict_adv, res_mask[:,1:])).float()
            adv_loss = nn.BCELoss()(next_predict_adv, next_true)

            kt_loss = pred_loss + ATKTConfig.BETA * adv_loss

            kt_loss.backward()

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

                predict, features = model(use_skill, use_ans)
                next_predict = (torch.masked_select(predict, res_mask[:,1:])).float()
                next_true = (torch.masked_select(use_ans[:,1:], res_mask[:,1:])).float()
                pred_loss = nn.BCELoss()(next_predict, next_true)

                total_loss.append(pred_loss.item())

                # 为了计算整体的acc，将该批次的长度以及该批次正确数添加到total_num和total_correct
                total_num += len(next_true)
                to_pred = (next_predict >= 0.5).long()
                total_correct += (next_true == to_pred).sum()

                # 为了计算整体的auc,将该批次的预测结果和真实标签添加到outputs和labels中
                labels.extend(next_true.view(-1).data.cpu().numpy())
                outputs.extend(next_predict.view(-1).data.cpu().numpy())

    loss = np.mean(total_loss)
    acc = total_correct.item() / total_num
    auc = metrics.roc_auc_score(labels, outputs)
    rmse = np.sqrt(metrics.mean_squared_error(y_true=labels, y_pred=outputs))
    return loss, acc, auc, rmse


def _l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)
