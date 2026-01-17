import torch
import matplotlib.pyplot as plt

from simplekt import simpleKT
from run import run_epoch
from config import BASICConfig, simpleKTConfig

if __name__ == '__main__':
    # 参数
    dataset = BASICConfig.DATASET
    pro_max = BASICConfig.N_PRO
    skill_max = BASICConfig.N_KNOWS
    device = BASICConfig.DEVICE

    model_name = simpleKTConfig.MODEL_NAME
    d_model = simpleKTConfig.D_MODEL
    n_blocks = simpleKTConfig.N_BLOCKS
    dropout = simpleKTConfig.DROPOUT
    d_ff = simpleKTConfig.D_FF
    max_seq = simpleKTConfig.SEQ_LEN
    kq_same = simpleKTConfig.KQ_SAME
    final_fc_dim = simpleKTConfig.FINAL_FC_DIM
    final_fc_dim2 = simpleKTConfig.FINAL_FC_DIM2
    num_attn_heads = simpleKTConfig.NUM_ATTN_HEADS
    separate_qa = simpleKTConfig.SEPARATE_QA
    l2 = simpleKTConfig.L2
    emb_type = simpleKTConfig.EMB_TYPE
    lr = simpleKTConfig.LEARNING_RATE
    epoch = simpleKTConfig.NUM_EPOCHS
    batch_size = simpleKTConfig.BATCH_SIZE
    min_seq = simpleKTConfig.MIN_SEQ_LEN

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

        model = simpleKT(skill_max, pro_max, d_model, n_blocks, dropout, d_ff, max_seq, kq_same, final_fc_dim,
                         final_fc_dim2, num_attn_heads, separate_qa, l2, emb_type).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

            # if early_stop_counter > patience:
            #     print(
            #         f"Early stopping triggered at epoch {epoch}, BEST_ACC: {best_acc:.4f}, BEST_AUC: {best_auc:.4f}, BEST_RMSE: {best_rmse:.4f}")
            #     break

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
