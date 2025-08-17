import torch

class BASICConfig:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # DATASET = 'assist09'
    # N_KNOWS = 167
    # N_PRO = 15925

    # DATASET = 'assist12'
    # N_KNOWS = 265
    # N_PRO = 53091

    # DATASET = 'assist17'
    # N_KNOWS = 102
    # N_PRO = 3162

    # DATASET = 'xes3g5m'
    # N_KNOWS = 1464
    # N_PRO = 7652

    # DATASET = 'junyi'
    # N_KNOWS = 39
    # N_PRO = 690

    # DATASET = 'eedi'
    # N_KNOWS = 1092
    # N_PRO = 27566

    DATASET = 'ednet'
    N_KNOWS = 1888
    N_PRO = 12192

class PKTConfig:
    BATCH_SIZE = 8

    MODEL_NAME = "PKT"
    MIN_SEQ_LEN = 2
    N_EPOCH = 600
    LR = 1e-4
    SZ_RNN_IN = 128
    SZ_RNN_OUT = 128
    N_RNN_LAYER = 1
    RNN_DROPOUT = 0.0
    L_FORWARD_PUNISH_THRESHOLD = 0.7
    L_BACKWARD_PUNISH_THRESHOLD = 0.3
    L_FORWARD_PUNISH = 1
    L_BACKWARD_PUNISH = 1
    G_PUNISH_THRESHOLD_COEF = 0.4
    S_PUNISH_THRESHOLD_COEF = 0.4
    EARLY_STOP = 30















