import torch


class BASICConfig:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # DATASET = 'assist12_1'
    # N_KNOWS = 265
    # N_PRO = 53091

    # DATASET = 'assist17'
    # N_KNOWS = 102
    # N_PRO = 3162

    DATASET = 'xes3g5m'
    N_KNOWS = 865
    N_PRO = 7652

    # DATASET = 'junyi'
    # N_KNOWS = 39
    # N_PRO = 690

    # DATASET = 'eedi'
    # N_KNOWS = 1092
    # N_PRO = 27566

    # DATASET = 'ednet'
    # N_KNOWS = 1888
    # N_PRO = 12192

    # assist09缺少间隔时间

class PSKTConfig:
    BATCH_SIZE = 64

    MODEL_NAME = "PSKT"
    SEQ_LEN = 100
    MIN_SEQ_LEN = 3
    NUM_EPOCH = 200
    LEARNING_RATE = 1e-3
    EMBED_DIM = 256
    CV_NUM = 0
    EARLY_STOP = 30

