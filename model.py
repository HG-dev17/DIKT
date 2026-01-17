import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerEncoder
from config import Config
import glo


class DIKT(nn.Module):
    def __init__(self, pro_max, skill_max, d_model, dropout):
        super(DIKT, self).__init__()
        self.pro_max = pro_max
        self.skill_max = skill_max
        self.d_model = d_model
        
        # 嵌入层
        self.problem_embed = nn.Embedding(pro_max, d_model)
        self.skill_embed = nn.Embedding(skill_max, d_model)
        self.answer_embed = nn.Embedding(2, d_model)  # 0和1两种答案
        
        # 时间特征处理
        self.time_linear = nn.Linear(1, d_model)
        
        # Transformer编码器
        self.encoders = nn.ModuleList([
            TransformerEncoder(
                n_heads=Config.HEADS,
                d_model=d_model,
                d_ff=d_model * 4,
                dropout=dropout
            ) for _ in range(Config.K_HOP)
        ])
        
        # 输出层
        self.out_linear = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, problem, answer, time):
        # problem: [batch_size, seq_len]
        # answer: [batch_size, seq_len] - 历史答案序列（用于预测下一个）
        # time: [batch_size, seq_len]
        
        batch_size, seq_len = problem.size()
        
        # 获取问题对应的技能
        pro2skill = glo.get_value('pro2skill')  # [pro_max, skill_max]
        skill_ids = torch.argmax(pro2skill[problem], dim=-1)  # [batch_size, seq_len]
        
        # 嵌入层
        problem_emb = self.problem_embed(problem)  # [batch_size, seq_len, d_model]
        skill_emb = self.skill_embed(skill_ids)    # [batch_size, seq_len, d_model]
        # 使用历史答案（向左shift，第一个位置用0填充）
        # 这样在位置i，我们使用的是位置i-1的答案来预测位置i
        answer_shifted = torch.cat([torch.zeros(batch_size, 1, dtype=answer.dtype, device=answer.device), 
                                    answer[:, :-1]], dim=1)
        answer_emb = self.answer_embed(answer_shifted)  # [batch_size, seq_len, d_model]
        
        # 时间特征处理
        time = time.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        time_emb = self.time_linear(time)  # [batch_size, seq_len, d_model]
        
        # 组合特征
        x = problem_emb + skill_emb + answer_emb + time_emb
        x = self.dropout(x)
        
        # Transformer编码
        for encoder in self.encoders:
            x = encoder(x, x, peek_cur=False)
        
        # 输出预测
        output = self.out_linear(x)  # [batch_size, seq_len, 1]
        output = torch.sigmoid(output).squeeze(-1)  # [batch_size, seq_len]
        
        return output
