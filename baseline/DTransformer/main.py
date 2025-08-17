import json
import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dtransformer import DTransformer
from run import run_epoch
from config import BASICConfig, DTRANSFORMERConfig

if __name__ == '__main__':
    # 参数
    dataset = BASICConfig.DATASET
    pro_max = BASICConfig.N_PRO
    skill_max = BASICConfig.N_KNOWS
    device = BASICConfig.DEVICE

    model_name = DTRANSFORMERConfig.MODEL_NAME
    d_model = DTRANSFORMERConfig.D_MODEL
    n_layers = DTRANSFORMERConfig.N_LAYERS
    n_heads = DTRANSFORMERConfig.N_HEADS
    n_know = DTRANSFORMERConfig.N_KNOW
    lambda_cl = DTRANSFORMERConfig.LAMBDA_CL
    dropout = DTRANSFORMERConfig.DROPOUT
    window = DTRANSFORMERConfig.WINDOW
    learning_rate = DTRANSFORMERConfig.LEARNING_RATE
    l2 = DTRANSFORMERConfig.L2
    epoch = DTRANSFORMERConfig.EPOCH
    patience = DTRANSFORMERConfig.EARLY_STOP
    batch_size = DTRANSFORMERConfig.BATCH_SIZE
    test_batch_size = DTRANSFORMERConfig.TEST_BATCH_SIZE
    min_seq=DTRANSFORMERConfig.MIN_SEQ
    max_seq=DTRANSFORMERConfig.MAX_SEQ

    ################################################## model training ##################################################
    avg_acc = 0
    avg_auc = 0
    avg_rmse = 0

    for fold in range(5):
        print(f"Starting fold {fold}...")

        # 读取数据
        mp2path = {
            dataset: {
                'train_path': f'../../pre_process_data/{dataset}/{fold}/train_test/train_question.txt',
                'test_path': f'../../pre_process_data/{dataset}/{fold}/train_test/test_question.txt',
            },
        }
        train_path = mp2path[dataset]['train_path']
        test_path = mp2path[dataset]['test_path']

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

        model = DTransformer(
           skill_max,
            pro_max,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_know=n_know,
            lambda_cl=lambda_cl,
            dropout=dropout,
            window=window,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2)
        criterion = nn.BCELoss()
        early_stop_counter = 0
        for epoch in range(epoch):
            train_loss, train_acc, train_auc, train_rmse = run_epoch(True, train_path, model, optimizer,
                                                                     batch_size, min_seq, max_seq, criterion, skill_max)
            print(
                f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}, train_rmse: {train_rmse:.4f}')
            res_train_loss.append(train_loss)
            res_train_acc.append(train_acc)
            res_train_auc.append(train_auc)
            res_train_rmse.append(train_rmse)

            test_loss, test_acc, test_auc, test_rmse = run_epoch(False, test_path, model, optimizer,
                                                                 test_batch_size, min_seq, max_seq, criterion, skill_max)
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
            else:
                early_stop_counter += 1

            if early_stop_counter > patience:
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

        # output = {
        #     "res_train_loss": res_train_loss,
        #     "res_train_acc": res_train_acc,
        #     "res_train_auc": res_train_auc,
        #     "res_train_rmse": res_train_rmse,
        #     "res_test_loss": res_test_loss,
        #     "res_test_acc": res_test_acc,
        #     "res_test_auc": res_test_auc,
        #     "res_test_rmse": res_test_rmse
        # }
        # result_summary = {
        #     "best_epoch": best_epoch,
        #     "best_auc": best_auc,
        #     "best_acc": best_acc,
        #     "best_rmse": best_rmse
        # }
        # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # file_name = f"GRKT{fold}_{timestamp}.json"
        # with open(os.path.join("../../output/GRKT", file_name), 'w') as w:
        #     result_summary["output"] = output  # 将训练和测试结果添加到result_summary中
        #     json.dump(result_summary, w, indent=4)

    avg_acc = avg_acc / 5
    avg_auc = avg_auc / 5
    avg_rmse = avg_rmse / 5

    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}, final_avg_rmse: {avg_rmse:.4f}')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
