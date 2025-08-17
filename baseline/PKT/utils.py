from sklearn.metrics import roc_curve, auc, accuracy_score
import torch
from torch import nn
from config import BASICConfig, PKTConfig

def selecting_mask(effLen, maxLen):
    bsz = effLen.size(0)
    mask = torch.arange(end=maxLen, device=BASICConfig.DEVICE).repeat(repeats=(bsz, 1)) < effLen.unsqueeze(1)
    return mask


def calculate_lossGlobal(predict, label, maskEffLen):
    predict = predict.masked_select(maskEffLen)
    label = label.masked_select(maskEffLen)

    loss_fn = nn.BCELoss(reduction='sum')
    bce_loss = loss_fn(predict, label)
    return bce_loss


def calculate_lossParams(params, nextSkill_oneHot, effLens):
    L, G, S= params
    lossList = []
    mask_L_forward = L[:, 1:, :] >= L[:, :-1, :]
    mask_L_forwardThres = (L[:, 1:, :] - L[:, :-1, :]) > PKTConfig.L_FORWARD_PUNISH_THRESHOLD
    lossL_forward = mask_L_forward * mask_L_forwardThres * (L[:, 1:, :] - L[:, :-1, :]) ** 2 * PKTConfig.L_FORWARD_PUNISH
    mask_L_backwardThres = (L[:, :-1, :]-L[:, 1:, :]) > PKTConfig.L_BACKWARD_PUNISH_THRESHOLD
    lossL_backward = ~mask_L_forward * mask_L_backwardThres * (L[:, :-1, :]-L[:, 1:, :]) ** 2 * PKTConfig.L_BACKWARD_PUNISH
    lossL = torch.sum(lossL_forward+lossL_backward)
    lossList.append(lossL)

    G = G * nextSkill_oneHot
    S = S * nextSkill_oneHot
    G_punishThreshold = effLens.type(torch.float) * PKTConfig.G_PUNISH_THRESHOLD_COEF
    S_punishThreshold = effLens.type(torch.float) * PKTConfig.S_PUNISH_THRESHOLD_COEF
    G_effSkillSum = torch.sum(G, dim=[1, 2])
    G_maskGt0 = (G_effSkillSum - G_punishThreshold) > 0
    G_effSkillSumPunish = (G_effSkillSum - G_punishThreshold) * G_maskGt0
    lossG = torch.sum(G_effSkillSumPunish)
    lossList.append(lossG)
    S_effSkillSum = torch.sum(S, dim=[1, 2])
    S_maskGt0 = (S_effSkillSum - S_punishThreshold) > 0
    S_effSkillSumPunish = (S_effSkillSum - S_punishThreshold) * S_maskGt0
    lossS = torch.sum(S_effSkillSumPunish)
    lossList.append(lossS)

    sumParamsLoss = 0
    for los in lossList:
        sumParamsLoss += los

    return sumParamsLoss, lossList
