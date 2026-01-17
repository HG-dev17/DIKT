import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

import glo
from model import DIKT
from run import run_epoch
from config import Config

if __name__ == '__main__':

    # 参数
    device = Config.DEVICE

    dataset = Config.DATASET
    skill_max = Config.NUM_SKILL + 1
    pro_max = Config.NUM_PROBLEM + 1

    model_name = Config.MODEL_NAME
    d_model = Config.DIM_HIDDEN
    batch_size = Config.BATCH_SIZE
    epochs = Config.EPOCHS
    dropout = Config.DROP_RATE
    learning_rate = Config.LEARNING_RATE
    min_seq = Config.MIN_SEQ
    max_seq = Config.MAX_SEQ
    patience = Config.EARLY_STOP

    grad_clip = 15.0

    ################################################## model training ##################################################
    avg_acc = 0
    avg_auc = 0
    avg_rmse = 0

    for fold in range(5):
        print(f"Starting fold {fold}...")

        # 读取数据
        mp2path = {
            dataset: {
                'train_path': f'pre_process_data/{dataset}/{fold}/train_test/train_question.txt',
                'test_path': f'pre_process_data/{dataset}/{fold}/train_test/test_question.txt',
                'ques_skill_path': f'pre_process_data/{dataset}/{fold}/graph/ques_skill.csv',
                'train_graph_path': f'pre_process_data/{dataset}/{fold}/graph/train_graphs.npz',
            },
        }
        train_path = mp2path[dataset]['train_path']
        test_path = mp2path[dataset]['test_path']
        ques_skill_path = mp2path[dataset]['ques_skill_path']
        train_graph_path = mp2path[dataset]['train_graph_path']

        # 存储到global_dict
        glo._init()

        pro2skill = torch.zeros((pro_max, skill_max)).to(device)
        for (x, y) in zip(pd.read_csv(ques_skill_path).values[:, 0],
                          pd.read_csv(ques_skill_path).values[:, 1]):
            pro2skill[x][y] = 1
        glo.set_value('pro2skill', pro2skill)

        glo.set_value('train_graph_path', train_graph_path)

        # 记录200epoch中最好的结果
        best_epoch = 0
        best_acc = 0
        best_auc = 0
        best_rmse = 0

        # 保存200epoch中所有结果
        res_train_loss = []
        res_train_acc = []
        res_train_auc = []
        res_train_rmse = []

        res_test_loss = []
        res_test_acc = []
        res_test_auc = []
        res_test_rmse = []

        criterion = nn.BCELoss()
        model = DIKT(pro_max, skill_max, d_model, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5, capturable=True)

        early_stop_counter = 0
        for epoch in range(epochs):

            train_loss, train_acc, train_auc, train_rmse = run_epoch(True, train_path, model, optimizer,
                                                                     batch_size, min_seq, max_seq, grad_clip, criterion)
            print(
                f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}, train_rmse: {train_rmse:.4f}')
            res_train_loss.append(train_loss)
            res_train_acc.append(train_acc)
            res_train_auc.append(train_auc)
            res_train_rmse.append(train_rmse)

            test_loss, test_acc, test_auc, test_rmse = run_epoch(False, test_path, model, optimizer,
                                                                 batch_size, min_seq, max_seq, grad_clip, criterion)
            print(
                f'epoch: {epoch}, test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}, test_rmse: {test_rmse:.4f}')
            res_test_loss.append(test_loss)
            res_test_acc.append(test_acc)
            res_test_auc.append(test_auc)
            res_test_rmse.append(test_rmse)

            if test_auc > best_auc:
                early_stop_counter = 0
                best_epoch = epoch
                best_auc = test_auc
                best_acc = test_acc
                best_rmse = test_rmse
                # 保存最好的模型
                best_model_path = f'./output/best_model_fold_{fold}.pt'
                torch.save(model.state_dict(), best_model_path)
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch}, BEST_ACC: {best_acc:.4f}, BEST_AUC: {best_auc:.4f}, BEST_RMSE: {best_rmse:.4f}")
                break

        print(f'*******************************************************************************')
        print(
            f'Fold {fold} completed, BEST_EPOCH: {best_epoch}, BEST_ACC: {best_acc:.4f}, BEST_AUC: {best_auc:.4f}, BEST_RMSE: {best_rmse:.4f}')
        print(f'*******************************************************************************')

        avg_acc += best_acc
        avg_auc += best_auc
        avg_rmse += best_rmse

        # 绘图并保存结果
        x = [i for i in range(len(res_train_loss))]
        plt.plot(x, res_train_loss, label='train_loss', color='blue', linestyle='dashed')
        plt.plot(x, res_train_acc, label='train_acc', color='green', linestyle='dashdot')
        plt.plot(x, res_train_auc, label='train_auc', color='red', linestyle='solid')
        plt.plot(x, res_train_rmse, label='train_rmse', color='pink', linestyle='dashed')
        plt.plot(x, res_test_loss, label='test_loss', color='yellow', linestyle='dashed')
        plt.plot(x, res_test_acc, label='test_acc', color='black', linestyle='dashdot')
        plt.plot(x, res_test_auc, label='test_auc', color='cyan', linestyle='solid')
        plt.plot(x, res_test_rmse, label='test_rmse', color='purple', linestyle='dashdot')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title(f'{model_name}-{dataset} Training and Testing Metrics over Epochs')
        plt.legend()
        plt.show()

    avg_acc = avg_acc / 5
    avg_auc = avg_auc / 5
    avg_rmse = avg_rmse / 5

    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*****************************************                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          **************************************')
    print(f'*******************************************************************************')
    print(f'final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}, final_avg_rmse: {avg_rmse:.4f}')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
