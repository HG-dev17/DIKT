import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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

    # Fold进度条
    fold_pbar = tqdm(range(5), desc='总体进度', ncols=100, position=0)
    for fold in fold_pbar:
        fold_pbar.set_description(f'Fold {fold}/4')
        print(f"\n{'='*80}")
        print(f"开始训练 Fold {fold}/4")
        print(f"{'='*80}")

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
        # capturable=True 只在CUDA设备上有效，CPU上会报错
        optimizer_kwargs = {'lr': learning_rate, 'weight_decay': 1e-5}
        if device.type == 'cuda':
            optimizer_kwargs['capturable'] = True
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)

        early_stop_counter = 0
        # Epoch进度条
        epoch_pbar = tqdm(range(epochs), desc=f'Fold {fold} Epoch进度', ncols=120, position=1, leave=True)
        for epoch in epoch_pbar:
            # 更新epoch进度条，显示当前正在训练
            epoch_pbar.set_description(f'Fold {fold} Epoch {epoch+1}/{epochs} [训练中]')
            
            train_loss, train_acc, train_auc, train_rmse = run_epoch(True, train_path, model, optimizer,
                                                                     batch_size, min_seq, max_seq, grad_clip, criterion)
            res_train_loss.append(train_loss)
            res_train_acc.append(train_acc)
            res_train_auc.append(train_auc)
            res_train_rmse.append(train_rmse)

            # 更新epoch进度条，显示当前正在验证
            epoch_pbar.set_description(f'Fold {fold} Epoch {epoch+1}/{epochs} [验证中]')
            
            test_loss, test_acc, test_auc, test_rmse = run_epoch(False, test_path, model, optimizer,
                                                                 batch_size, min_seq, max_seq, grad_clip, criterion)
            res_test_loss.append(test_loss)
            res_test_acc.append(test_acc)
            res_test_auc.append(test_auc)
            res_test_rmse.append(test_rmse)

            # 更新进度条显示
            epoch_pbar.set_description(f'Fold {fold} Epoch {epoch+1}/{epochs} [完成]')
            epoch_pbar.set_postfix({
                'Train': f'Loss:{train_loss:.3f} Acc:{train_acc:.3f} AUC:{train_auc:.3f}',
                'Test': f'Loss:{test_loss:.3f} Acc:{test_acc:.3f} AUC:{test_auc:.3f}',
                'Best': f'AUC:{best_auc:.3f}'
            })

            if test_auc > best_auc:
                early_stop_counter = 0
                best_epoch = epoch
                best_auc = test_auc
                best_acc = test_acc
                best_rmse = test_rmse
                # 保存最好的模型
                import os
                os.makedirs('output', exist_ok=True)
                best_model_path = f'./output/best_model_fold_{fold}.pt'
                torch.save(model.state_dict(), best_model_path)
                epoch_pbar.set_postfix({
                    'Train': f'Loss:{train_loss:.3f} Acc:{train_acc:.3f} AUC:{train_auc:.3f}',
                    'Test': f'Loss:{test_loss:.3f} Acc:{test_acc:.3f} AUC:{test_auc:.3f}',
                    'Best': f'AUC:{best_auc:.3f} ✓'
                })
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                epoch_pbar.set_postfix({
                    'Status': 'Early Stop',
                    'Best': f'ACC:{best_acc:.3f} AUC:{best_auc:.3f} RMSE:{best_rmse:.3f}'
                })
                print(f"\n早停触发于 epoch {epoch}, 最佳结果: ACC={best_acc:.4f}, AUC={best_auc:.4f}, RMSE={best_rmse:.4f}")
                break
        
        epoch_pbar.close()

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
