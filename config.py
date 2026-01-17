import torch

class Config:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    DATASET = 'assist09'
    NUM_SKILL = 150  # 更新为实际数据中的技能数，原123
    NUM_PROBLEM = 26688  # 更新为实际数据中的题目数，原15925
    BATCH_SIZE = 64

    # DATASET = 'assist12_1'
    # NUM_SKILL = 265
    # NUM_PROBLEM = 53091
    # BATCH_SIZE = 50

    # DATASET = 'assist17'
    # NUM_SKILL = 102
    # NUM_PROBLEM = 3162
    # BATCH_SIZE = 64

    # DATASET = 'junyi'
    # NUM_SKILL = 39
    # NUM_PROBLEM = 690
    # BATCH_SIZE = 128

    # DATASET = 'eedi'
    # NUM_SKILL = 309
    # NUM_PROBLEM = 27566
    # BATCH_SIZE = 64

    # DATASET = 'ednet'
    # NUM_SKILL = 188
    # NUM_PROBLEM = 12192
    # BATCH_SIZE = 64

    # DATASET = 'xes3g5m'
    # NUM_SKILL = 865
    # NUM_PROBLEM = 7652
    # BATCH_SIZE = 15

    MODEL_NAME = "DIKT"

    # 模型维度
    DIM_HIDDEN = 64
    SKILL_HIDDEN = 64
    DROP_RATE = 0.1
    LEARNING_RATE = 0.001
    HEADS = 8
    K_HOP = 3
    MAX_TIME_GAP = 86400

    MIN_SEQ = 3
    MAX_SEQ = 200
    EPOCHS = 2 #200
    EARLY_STOP = 20









