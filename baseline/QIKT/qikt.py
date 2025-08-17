import torch
import torch.nn as nn
import torch.nn.functional as F
from config import QIKTConfig


class MLP(nn.Module):
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()

        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p=dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))


def get_outputs(self, emb_qc_shift, h, q, c, model_type='question'):
    outputs = {}

    if model_type == 'question':
        h_next = torch.cat([emb_qc_shift, h], axis=-1)

        y_question_next = torch.sigmoid(self.out_question_next(h_next))  # [batch_size, max_seq-1, 1]
        y_question_all = torch.sigmoid(self.out_question_all(h))  # [batch_size, max_seq-1, num_q]
        outputs["y_question_next"] = y_question_next.squeeze(-1)  # [batch_size, max_seq-1]
        outputs["y_question_all"] = (y_question_all * F.one_hot(q[:, 1:], self.num_q)).sum(
            -1)  # [batch_size, max_seq-1]
    else:
        h_next = torch.cat([emb_qc_shift, h], axis=-1)
        y_concept_next = torch.sigmoid(self.out_concept_next(h_next))
        y_concept_all = torch.sigmoid(self.out_concept_all(h))
        outputs["y_concept_next"] = self.get_avg_fusion_concepts(y_concept_next, c[:, 1:, :])
        outputs["y_concept_all"] = self.get_avg_fusion_concepts(y_concept_all, c[:, 1:, :])

    return outputs


class QIKTNet(nn.Module):
    def __init__(self, num_q, num_c, emb_size, dropout, mlp_layer_num):
        super().__init__()
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.mlp_layer_num = mlp_layer_num

        self.concept_emb = nn.Parameter(torch.randn(self.num_c, self.emb_size), requires_grad=True)
        self.que_emb = nn.Embedding(self.num_q, self.emb_size)

        self.que_lstm_layer = nn.LSTM(self.emb_size * 4, self.hidden_size, batch_first=True)

        self.dropout_layer = nn.Dropout(dropout)

        self.out_question_next = MLP(self.mlp_layer_num, self.hidden_size * 3, 1, dropout)
        self.out_question_all = MLP(self.mlp_layer_num, self.hidden_size, num_q, dropout)

        self.concept_lstm_layer = nn.LSTM(self.emb_size * 2, self.hidden_size, batch_first=True)

        self.out_concept_next = MLP(self.mlp_layer_num, self.hidden_size * 3, num_c, dropout)
        self.out_concept_all = MLP(self.mlp_layer_num, self.hidden_size, num_c, dropout)

    def get_avg_fusion_concepts(self, y_concept, cshft):
        """获取知识点 fusion 的预测结果
        """
        # y_concept batch_size,max_len-1,num_c

        max_num_concept = cshft.shape[-1]
        concept_mask = torch.where(cshft.long() == 0, False, True) # batch_size,max_len-1,4
        concept_index = F.one_hot(cshft, self.num_c) # batch_size,max_len-1,4,num_c
        concept_sum = (y_concept.unsqueeze(2).repeat(1, 1, max_num_concept, 1) * concept_index).sum(-1) # batch_size,max_len-1,4
        concept_sum = concept_sum * concept_mask  # batch_size,max_len-1,4
        y_concept = concept_sum.sum(-1) / torch.where(concept_mask.sum(-1) != 0, concept_mask.sum(-1), 1)  # batch_size,max_len-1
        return y_concept

    def get_avg_skill_emb(self, c):
        # [batch_size, seq_len, emb_dim]
        concept_emb_sum = self.concept_emb[c, :].sum(axis=-2)

        # [batch_size, seq_len, 1]
        concept_num = torch.where(c != 0, 1, 0).sum(axis=-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num)

        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg

    def forward(self, q, c, r):
        emb_c = self.get_avg_skill_emb(c)  # [batch, max_len, emb_size]
        emb_q = self.que_emb(q)  # [batch, max_len, emb_size]
        emb_qc = torch.cat([emb_q, emb_c], dim=-1)  # [batch, max_len, 2*emb_size]

        # [batch, max_len, 4*emb_size]
        emb_qca = torch.cat([emb_qc.mul((1 - r).unsqueeze(-1).repeat(1, 1, self.emb_size * 2)),
                             emb_qc.mul((r).unsqueeze(-1).repeat(1, 1, self.emb_size * 2))], dim=-1)
        emb_qc_shift = emb_qc[:, 1:, :]
        emb_qca_current = emb_qca[:, :-1, :]

        # question model
        que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current)[0])
        que_outputs = get_outputs(self, emb_qc_shift, que_h, q, c, model_type="question")
        outputs = que_outputs

        # concept model
        emb_ca = torch.cat([emb_c.mul((1 - r).unsqueeze(-1).repeat(1, 1, self.emb_size)),
                            emb_c.mul((r).unsqueeze(-1).repeat(1, 1, self.emb_size))], dim=-1)
        emb_ca_current = emb_ca[:, :-1, :]

        concept_h = self.dropout_layer(self.concept_lstm_layer(emb_ca_current)[0])
        concept_outputs = get_outputs(self, emb_qc_shift, concept_h, q, c, model_type="concept")
        outputs['y_concept_all'] = concept_outputs['y_concept_all']
        outputs['y_concept_next'] = concept_outputs['y_concept_next']

        return outputs


class QIKT(nn.Module):
    def __init__(self, num_q, num_c, emb_size, dropout, mlp_layer_num, output_mode):

        super().__init__()

        self.model = QIKTNet(num_q=num_q, num_c=num_c, emb_size=emb_size, dropout=dropout, mlp_layer_num=mlp_layer_num)

        self.output_mode = output_mode

    def forward(self, use_problem, use_skill, use_ans):

        outputs = self.model(use_problem, use_skill, use_ans)

        output_c_all_lambda = QIKTConfig.OUTPUT_C_ALL_LAMBDA
        output_c_next_lambda = QIKTConfig.OUTPUT_C_NEXT_LAMBDA
        output_q_all_lambda = QIKTConfig.OUTPUT_Q_ALL_LAMBDA
        output_q_next_lambda = QIKTConfig.OUTPUT_Q_NEXT_LAMBDA

        if self.output_mode == "an_irt":
            def sigmoid_inverse(x, epsilon=1e-8):
                return torch.log(x / (1 - x + epsilon) + epsilon)

            y = sigmoid_inverse(outputs['y_question_all']) * output_q_all_lambda + sigmoid_inverse(
                outputs['y_concept_all']) * output_c_all_lambda + sigmoid_inverse(
                outputs['y_concept_next']) * output_c_next_lambda
            y = torch.sigmoid(y)
        else:
            # output weight
            y = outputs['y_question_all'] * output_q_all_lambda + outputs['y_concept_all'] * output_c_all_lambda + \
                outputs['y_concept_next'] * output_c_next_lambda
            y = y / (output_q_all_lambda + output_c_all_lambda + output_c_next_lambda)
        outputs['y'] = y

        return outputs
