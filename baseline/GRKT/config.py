import torch

class BASICConfig:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

    DATASET = 'xes3g5m'
    N_KNOWS = 865
    N_PRO = 7652
    MAX_NUM_PRO_SKILL = 6

    # DATASET = 'junyi'
    # N_KNOWS = 39
    # N_PRO = 690
    # MAX_NUM_PRO_SKILL = 1

    # DATASET = 'eedi'
    # N_KNOWS = 309
    # N_PRO = 27566
    # MAX_NUM_PRO_SKILL = 6

    # DATASET = 'ednet'
    # N_KNOWS = 188
    # N_PRO = 12192
    # MAX_NUM_PRO_SKILL = 7

class GRKTConfig:
    BATCH_SIZE = 48

    MODEL_NAME = "GRKT"
    D_HIDDEN = 128
    K_HIDDEN = 16
    POS_MODE = 'sigmoid'
    K_HOP = 1
    THRESH = 0.6
    TAU = 0.2
    ALPHA = 0.01
    LEARNING_RATE = 0.001
    L2 = 0.000
    EPOCH = 500
    MIN_SEQ = 10
    MAX_SEQ = 100
    EARLY_STOP = 30




