import torch

class BASICConfig:
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
class DTRANSFORMERConfig:
    BATCH_SIZE = 16

    MODEL_NAME = "DTRANSFORMER"
    D_MODEL = 128
    N_LAYERS = 3
    N_HEADS = 8
    N_KNOW = 32
    LAMBDA_CL = 0.1
    DROPOUT = 0.2
    WINDOW = 1
    LEARNING_RATE = 1e-3
    L2 = 1e-5
    EPOCH = 100
    EARLY_STOP = 10
    TEST_BATCH_SIZE = 32
    MIN_SEQ = 5
    MAX_SEQ = 200



