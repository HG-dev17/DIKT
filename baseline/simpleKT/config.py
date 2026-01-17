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

class simpleKTConfig:
   BATCH_SIZE = 64

   MODEL_NAME = "simpleKT"
   D_MODEL = 256
   N_BLOCKS = 2
   DROPOUT = 0.1
   D_FF = 256
   MIN_SEQ_LEN = 3
   SEQ_LEN = 200
   KQ_SAME = 1
   FINAL_FC_DIM = 256
   FINAL_FC_DIM2 = 256
   NUM_ATTN_HEADS = 4
   SEPARATE_QA = False
   L2 = 1e-5
   EMB_TYPE = "qid"
   LEARNING_RATE = 1e-4
   NUM_EPOCHS = 200











