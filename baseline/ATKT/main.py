import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from atkt import KT_backbone
from run import run_epoch
from config import BASICConfig, ATKTConfig

if __name__ == '__main__':
    # 参数
    dataset = BASICConfig.DATASET
    pro_max = BASICConfig.N_PRO
    skill_max = BASICConfig.N_KNOWS
    device = BASICConfig.DEVICE

    model_name = ATKTConfig.MODEL_NAME
    skill_emb_dim = ATKTConfig.SKILL_EMB_DIM
    answer_emb_dim = ATKTConfig.ANSWER_EMB_DIM
    hidden_emb_dim = ATKTConfig.HIDDEN_EMB_DIM
    lr = ATKTConfig.LR
    lr_decay = ATKTConfig.LR_DECAY
    gamma = ATKTConfig.GAMMA
    patience = ATKTConfig.EARLY_STOP
    epoch = ATKTConfig.MAX_ITER
    batch_size = ATKTConfig.BATCH_SIZE
    max_seq = ATKTConfig.SEQLEN
    min_seq = ATKTConfig.MIN_SEQ

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
        min_test_loss = 0

        # 保存200epoch中所有结果
        res_train_loss = []
        res_train_acc = []
        res_train_auc = []
        res_train_rmse = []

        res_test_loss = []
        res_test_acc = []
        res_test_auc = []
        res_test_rmse = []

        model = KT_backbone(skill_emb_dim, answer_emb_dim, hidden_emb_dim, skill_max).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)

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

            test_loss, test_acc, test_auc, test_rmse = run_epoch(False, test_path, model, optimizer,
                                                                 batch_size, min_seq, max_seq)
            print(
                f'epoch: {epoch}, test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}, test_rmse: {test_rmse:.4f}')
            scheduler.step()

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
