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


class CL4KTConfig:
    BATCH_SIZE = 256

    MODEL_NAME = "CL4KT"
    REG_CL = 0.1
    MASK_PROB = 0.2
    CROP_PROB = 0.3
    PERMUTE_PROB = 0.3
    REPLACE_PROB = 0.3
    NEGATIVE_PROB = 1.0
    DROPOUT = 0.2
    L2 = 0.0
    LR = 0.001
    HIDDEN_SIZE = 64
    NUM_BLOCKS = 2
    NUM_ATTN_HEADS = 8
    KQ_SAME = True
    FINAL_FC_DIM = 512
    D_FF = 1024
    TEMP = 0.05
    HARD_NEGATIVE_WEIGHT = 1.0


    EVAL_BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    SEQ_LEN = 100
    NUM_EPOCHS = 300
    MAX_GRAD_NORM = 2.0
    EARLY_STOP = 10
