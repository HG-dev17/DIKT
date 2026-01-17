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


class AKTConfig:
    BATCH_SIZE = 64

    MODEL_NAME = "AKT"
    N_BLOCK = 1
    D_MODEL = 256
    DROPOUT = 0.05
    KQ_SAME = 1
    L2 = 1e-5
    LR = 1e-5
    MODEL_TYPE = 'akt'
    MAX_SEQ = 200
    MIN_SEQ = 0
    EARLY_STOP = 40
    EPOCH = 300
    MAX_GRAD_ORM = -1
    N_HEADS = 8
