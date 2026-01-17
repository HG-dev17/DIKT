import torch

class BASICConfig:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # DATASET = 'assist12_1'
    # N_KNOWS = 265
    # N_PRO = 53091

    # DATASET = 'assist17'
    # N_KNOWS = 102
    # N_PRO = 3162

    # DATASET = 'junyi'
    # N_KNOWS = 39
    # N_PRO = 690

    DATASET = 'ednet'
    N_KNOWS = 1888
    N_PRO = 12192

    # xes3g5m和eddi缺少响应时间
    # assistment09缺少间隔时间

class LPKTConfig:
    BATCH_SIZE = 6

    MODEL_NAME = "LPKT"
    SEQ_LEN =  500
    MIN_SEQ_LEN = 2
    D_K = 128
    D_A = 50
    D_E = 128
    Q_GAMMA = 0.03
    DROP0UT = 0.2
    LEARNING_RATE = 0.003
    LR_DECAY_STEP = 10
    LR_DECAY_RATE = 0.5
    NUM_EPOCH = 30


