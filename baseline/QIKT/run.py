import torch
from tqdm import tqdm
import numpy as np
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics
from load_data import getLoader
from config import QIKTConfig


def run_epoch(is_train, path, model, optimizer, batch_size, min_problem_num,
              max_problem_num):
    total_loss = []
    total_correct = 0
    total_num = 0
    labels = []
    model_outputs = []

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
            outputs = model(use_problem, use_skill, use_ans)

            # 计算损失
            y_question_all = torch.masked_select(outputs['y_question_all'], res_mask[:,1:])
            y_concept_all = torch.masked_select(outputs['y_concept_all'], res_mask[:,1:])
            y_question_next = torch.masked_select(outputs['y_question_next'], res_mask[:,1:])
            y_concept_next = torch.masked_select(outputs['y_concept_next'], res_mask[:,1:])
            next_predict = torch.masked_select(outputs['y'], res_mask[:,1:])
            next_true = torch.masked_select(use_ans, res_mask)

            loss_q_all = binary_cross_entropy(y_question_all, next_true.float())
            loss_c_all = binary_cross_entropy(y_concept_all, next_true.float())
            loss_q_next = binary_cross_entropy(y_question_next, next_true.float())
            loss_c_next = binary_cross_entropy(y_concept_next, next_true.float())
            loss_kt = binary_cross_entropy(next_predict, next_true.float())

            loss = loss_kt + QIKTConfig.LOSS_Q_ALL_LAMBDA * loss_q_all + QIKTConfig.OUTPUT_C_ALL_LAMBDA * loss_c_all + QIKTConfig.LOSS_C_NEXT_LAMBDA * loss_c_next + QIKTConfig.OUTPUT_Q_NEXT_LAMBDA * loss_q_next

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # total_loss存放着该批次的损失值，目的是为了对所有批次的损失值求平均
            total_loss.append(loss.item())

            # 为了计算整体的acc，将该批次的长度以及该批次正确数添加到total_num和total_correct
            total_num += len(next_true)
            to_pred = (next_predict >= 0.5).long()
            total_correct += (next_true == to_pred).sum()

            # 为了计算整体的auc,将该批次的预测结果和真实标签添加到outputs和labels中
            labels.extend(next_true.view(-1).data.cpu().numpy())
            model_outputs.extend(next_predict.view(-1).data.cpu().numpy())
        else:
            with torch.no_grad():

                # 前向传播
                outputs = model(use_problem, use_skill, use_ans)

                # 计算损失
                y_question_all = torch.masked_select(outputs['y_question_all'], res_mask[:,1:])
                y_concept_all = torch.masked_select(outputs['y_concept_all'], res_mask[:,1:])
                y_question_next = torch.masked_select(outputs['y_question_next'], res_mask[:,1:])
                y_concept_next = torch.masked_select(outputs['y_concept_next'], res_mask[:,1:])
                next_predict = torch.masked_select(outputs['y'], res_mask[:,1:])
                next_true = torch.masked_select(use_ans, res_mask)

                loss_q_all = binary_cross_entropy(y_question_all, next_true.float())
                loss_c_all = binary_cross_entropy(y_concept_all, next_true.float())
                loss_q_next = binary_cross_entropy(y_question_next, next_true.float())
                loss_c_next = binary_cross_entropy(y_concept_next, next_true.float())
                loss_kt = binary_cross_entropy(next_predict, next_true.float())

                loss = loss_kt + QIKTConfig.LOSS_Q_ALL_LAMBDA * loss_q_all + QIKTConfig.OUTPUT_C_ALL_LAMBDA * loss_c_all + QIKTConfig.LOSS_C_NEXT_LAMBDA * loss_c_next + QIKTConfig.OUTPUT_Q_NEXT_LAMBDA * loss_q_next

                # total_loss存放着该批次的损失值，目的是为了对所有批次的损失值求平均
                total_loss.append(loss.item())

                # 为了计算整体的acc，将该批次的长度以及该批次正确数添加到total_num和total_correct
                total_num += len(next_true)
                to_pred = (next_predict >= 0.5).long()
                total_correct += (next_true == to_pred).sum()

                # 为了计算整体的auc,将该批次的预测结果和真实标签添加到outputs和labels中
                labels.extend(next_true.view(-1).data.cpu().numpy())
                model_outputs.extend(next_predict.view(-1).data.cpu().numpy())

    loss = np.mean(total_loss)
    acc = total_correct.item() / total_num
    auc = metrics.roc_auc_score(labels, model_outputs)
    rmse = np.sqrt(metrics.mean_squared_error(y_true=labels, y_pred=model_outputs))
    return loss, acc, auc, rmse
