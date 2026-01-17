import torch


class BASICConfig:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # DATASET = 'assist09'
    # N_KNOWS = 167
    # N_PRO = 15925

    #DATASET = 'assist12'
    #N_KNOWS = 265
    #N_PRO = 53091

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


class REKTConfig:
    BATCH_SIZE = 64

    MODEL_NAME = "REKT"
    DROPOUT = 0.4
    D = 128
    LEARNING_RATE = 0.002
    NUM_EPOCHS = 200
    SEQ_LEN = 200
    MIN_SEQ_LEN = 3
