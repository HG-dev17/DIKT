import torch
from tqdm import tqdm
import numpy as np
from sklearn import metrics

from load_data import getLoader
from utils import selecting_mask, calculate_lossGlobal, calculate_lossParams


def run_epoch(is_train, path, model, optimizer, batch_size, n_question, n_skill, device):
    total_loss = []
    total_correct = 0
    total_num = 0
    labels = []
    outputs = []

    if is_train:
        model.train()
    else:
        model.eval()

    data_loader = getLoader(is_train, path, batch_size, n_question, n_skill)

    # 会把数据集分为多个batch_size大小的数据（i表示数据的个数）
    # 调用load_data中的getitem方法取data
    for n_batch, batch in tqdm(enumerate(data_loader), desc='加载中...', ncols=100):
        effLens_batch, \
            currQuestionAddLabel_batch, currQuestionID_batch, \
            currSkillAddLabel_batch, currSkillID_batch, currSkill_oneHot_batch, \
            currLabel_batch, \
            nextQuestionID_batch, \
            nextSkillID_batch, nextSkill_oneHot_batch, \
            nextLabel_batch = batch

        effLens_batch, \
            currQuestionAddLabel_batch, \
            currSkillAddLabel_batch, \
            nextQuestionID_batch, \
            nextSkillID_batch, nextSkill_oneHot_batch, \
            nextLabel_batch = effLens_batch.to(device), \
            currQuestionAddLabel_batch.to(device), \
            currSkillAddLabel_batch.to(device), \
            nextQuestionID_batch.to(device), \
            nextSkillID_batch.to(device), nextSkill_oneHot_batch.to(device), \
            nextLabel_batch.to(device)

        bsz, maxLen_batch = currSkillAddLabel_batch.size()
        maskEffLen_batch = selecting_mask(effLen=effLens_batch, maxLen=maxLen_batch)

        if is_train:
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            predict_batch, predictAllSkill_batch, [L_skill, G, S] = model.forward(currSkillAddLabel_batch,
                                                                                  currQuestionAddLabel_batch,
                                                                                  nextSkill_oneHot_batch,
                                                                                  nextSkillID_batch,
                                                                                  nextQuestionID_batch)

            next_predict = (torch.masked_select(predict_batch, maskEffLen_batch)).float()
            next_true = (torch.masked_select(nextLabel_batch, maskEffLen_batch)).float()

            lossD_batch = calculate_lossGlobal(predict_batch, nextLabel_batch, maskEffLen_batch)
            lossParam_batch, lossList = calculate_lossParams([L_skill, G, S], nextSkill_oneHot_batch, effLens_batch)
            (lossD_batch + lossParam_batch).backward()

            # 更新参数
            optimizer.step()

            # total_loss存放着该批次的损失值，目的是为了对所有批次的损失值求平均
            total_loss.append(lossD_batch.item())

            # 为了计算整体的acc，将该批次的长度以及该批次正确数添加到total_num和total_correct
            total_num += len(next_true)
            to_pred = (next_predict >= 0.5).long()
            total_correct += (next_true == to_pred).sum()

            # 为了计算整体的auc,将该批次的预测结果和真实标签添加到outputs和labels中
            labels.extend(next_true.view(-1).data.cpu().numpy())
            outputs.extend(next_predict.view(-1).data.cpu().numpy())
        else:
            with torch.no_grad():
                predict_batch, predictAllSkill_batch, _ = model.forward(currSkillAddLabel_batch,
                                                                        currQuestionAddLabel_batch,
                                                                        nextSkill_oneHot_batch,
                                                                        nextSkillID_batch,
                                                                        nextQuestionID_batch)

                next_predict = (torch.masked_select(predict_batch, maskEffLen_batch)).float()
                next_true = (torch.masked_select(nextLabel_batch, maskEffLen_batch)).float()

                lossD_batch = calculate_lossGlobal(predict_batch, nextLabel_batch, maskEffLen_batch)
                total_loss.append(lossD_batch.item())

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
