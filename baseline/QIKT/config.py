import torch


class BASICConfig:
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # DATASET = 'assist09'
    # N_KNOWS = 123
    # N_PRO = 15925
    # MAX_NUM_PRO_SKILL = 4

    # DATASET = 'assist12'
    # N_KNOWS = 265
    # N_PRO = 53091
    # MAX_NUM_PRO_SKILL = 1

    # DATASET = 'assist17'
    # N_KNOWS = 102
    # N_PRO = 3162
    # MAX_NUM_PRO_SKILL = 1

    # DATASET = 'xes3g5m'
    # N_KNOWS = 865
    # N_PRO = 7652
    # MAX_NUM_PRO_SKILL = 6

    # DATASET = 'junyi'
    # N_KNOWS = 39
    # N_PRO = 690
    # MAX_NUM_PRO_SKILL = 1

    # DATASET = 'eedi'
    # N_KNOWS = 309
    # N_PRO = 27566
    # MAX_NUM_PRO_SKILL = 6

    DATASET = 'ednet'
    N_KNOWS = 188
    N_PRO = 12192
    MAX_NUM_PRO_SKILL = 7

class QIKTConfig:
    BATCH_SIZE = 64

    MODEL_NAME = "QIKT"
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 200
    DROPOUT = 0.4
    EMB_SIZE = 300
    MLP_LAYER_NUM = 2
    LOSS_Q_ALL_LAMBDA = 0
    LOSS_C_ALL_LAMBDA = 0
    LOSS_Q_NEXT_LAMBDA = 0
    LOSS_C_NEXT_LAMBDA = 0
    OUTPUT_Q_ALL_LAMBDA = 1
    OUTPUT_C_ALL_LAMBDA = 1
    OUTPUT_Q_NEXT_LAMBDA = 0
    OUTPUT_C_NEXT_LAMBDA = 1
    OUTPUT_MODE = "an_irt"
    SEQ_LEN = 200
    MIN_SEQ_LEN = 3



