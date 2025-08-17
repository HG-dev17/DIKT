import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

from lpkt import LPKT
from run import run_epoch
from config import BASICConfig, LPKTConfig

if __name__ == '__main__':
    # 参数
    dataset = BASICConfig.DATASET
    pro_max = BASICConfig.N_PRO
    skill_max = BASICConfig.N_KNOWS
    device = BASICConfig.DEVICE

    model_name = LPKTConfig.MODEL_NAME
    gamma = LPKTConfig.Q_GAMMA
    d_a = LPKTConfig.D_A
    d_e = LPKTConfig.D_E
    d_k = LPKTConfig.D_K
    batch_size = LPKTConfig.BATCH_SIZE
    dropout = LPKTConfig.DROP0UT
    lr = LPKTConfig.LEARNING_RATE
    lr_decay_step = LPKTConfig.LR_DECAY_STEP
    lr_decay_rate = LPKTConfig.LR_DECAY_RATE
    epoch = LPKTConfig.NUM_EPOCH
    max_seq = LPKTConfig.SEQ_LEN
    min_seq = LPKTConfig.MIN_SEQ_LEN

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
                'ques_skill_path': f'../../pre_process_data/{dataset}/{fold}/graph/ques_skill.csv',
            },
        }
        train_path = mp2path[dataset]['train_path']
        test_path = mp2path[dataset]['test_path']
        ques_skill_path = mp2path[dataset]['ques_skill_path']

        pro2skill = torch.full((pro_max + 1, skill_max + 1), gamma).to(device)
        for (x, y) in zip(pd.read_csv(ques_skill_path).values[:, 0],
                          pd.read_csv(ques_skill_path).values[:, 1]):
            pro2skill[x][y] = 1

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

        model = LPKT(pro_max, skill_max, d_a, d_e, d_k, pro2skill, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8, betas=(0.1, 0.999), weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)

        early_stop_counter = 0
        for epoch in range(epoch):
            train_loss, train_acc, train_auc, train_rmse = run_epoch(True, train_path, model, optimizer,
                                                                     batch_size, min_seq, max_seq)
            print(
                f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}, train_rmse: {train_rmse:.4f}')
            res_train_loss.append(train_loss)
            res_train_acc.append(train_acc)
            res_train_auc.append(train_auc)
            res_train_rmse.append(train_rmse)

            scheduler.step()

            test_loss, test_acc, test_auc, test_rmse = run_epoch(False, test_path, model, optimizer,
                                                                 batch_size, min_seq, max_seq)
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

            if early_stop_counter > 10:
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
