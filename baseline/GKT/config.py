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

    DATASET = 'xes3g5m'
    N_KNOWS = 1464
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


class GKTConfig:
    BATCH_SIZE = 8

    MODEL_NAME = "GKT"
    LEARNING_RATE = 1e-3
    DROPOUT = 0.5
    HIDDEN_DIM = 64
    EMB_SIZE = 64
    GRAPH_TYPE = 'transition'
    MAX_SEQ = 200
    MIN_SEQ = 3
    EARLY_STOP = 30
    EPOCH = 200

