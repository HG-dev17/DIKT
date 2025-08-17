import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding

from config import Config


# 计算注意力分数
def attention(q, k, v, mask, pos_bias, dropout, gamma=None):
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()

    # if pos_bias is not None:
    #     scores += pos_bias

    if gamma is not None:
        x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(Config.DEVICE)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)
            scores_ = scores_ * mask.float()

            distcum_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_effect = torch.abs(x1 - x2)[None, None, :, :]
            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()

        softplus = nn.Softplus()
        gamma = -1.0 * softplus(gamma).unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

        scores *= total_effect

    if pos_bias is not None:
        scores += pos_bias

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = scores * mask.float()
    scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output, scores


# 下三角掩码
def get_combined_mask(query, key, peek_cur=False):
    seqlen = query.size(1)

    # 下三角矩阵掩码（Causal Mask）
    causal_mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1).bool()
    causal_mask = causal_mask[None, None, :, :].to(Config.DEVICE)

    return causal_mask


class ALiBiPositionalEmbeddings(nn.Module):
    def __init__(self, attn_heads, max_seq):
        super().__init__()
        self.attn_heads = attn_heads
        self.max_len = max_seq

        self.slopes = torch.Tensor(self.get_slopes(attn_heads)).unsqueeze(1).unsqueeze(1)  # attn_heads, 1, 1

        self.alibi = self.slopes * torch.arange(self.max_len).unsqueeze(0).unsqueeze(0).expand(self.attn_heads, -1,
                                                                                               -1)  # (attn_heads, 1, max_len)

    # 返回长度为n的斜率列表
    def get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + self.get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    def buffered_future_mask(self, tensor):
        _future_mask = torch.zeros([self.max_len, self.max_len]).unsqueeze(0).unsqueeze(0)
        dim = tensor.size(2)
        _future_mask = _future_mask + self.alibi.unsqueeze(0).repeat(tensor.shape[0], 1, 1, 1)
        _future_mask = _future_mask.to(tensor.device)
        return _future_mask[:tensor.shape[0] * self.attn_heads, :dim, :dim]


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=Config.DROP_RATE, kq_same=True, bias=True, rope_v=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.kq_same = kq_same
        self.bias = bias
        self.rope_v = rope_v

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self.rope = RotaryEmbedding(dim=self.d_k) if self.rope_v else None

        self.pos_bias = ALiBiPositionalEmbeddings(n_heads, Config.MAX_SEQ)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.q_linear.weight)
        torch.nn.init.xavier_uniform_(self.v_linear.weight)
        if not self.kq_same:
            torch.nn.init.xavier_uniform_(self.k_linear.weight)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)

        if self.bias:
            torch.nn.init.constant_(self.q_linear.bias, 0.)
            torch.nn.init.constant_(self.v_linear.bias, 0.)
            if not self.kq_same:
                torch.nn.init.constant_(self.k_linear.bias, 0.)
            torch.nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask):
        bs = q.size(0)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        if self.rope_v:
            freqs = self.rope(torch.arange(Config.MAX_SEQ).to(Config.DEVICE))
            freqs = freqs[:Config.MAX_SEQ]
            freqs = freqs[None, None, ...]
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        pos_bias = self.pos_bias.buffered_future_mask(q)
        output, scores = attention(
            q,
            k,
            v,
            mask,
            pos_bias,
            self.dropout,
            self.gammas,
        )

        concat = output.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        output = self.out_proj(concat)
        return output, scores


# 前馈神经网络
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=Config.DROP_RATE):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout, max_seq=Config.MAX_SEQ):
        super().__init__()
        self.max_seq = max_seq
        self.self_attention = MultiHeadAttention(n_heads, d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-8)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-8)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, value, peek_cur):
        mask = get_combined_mask(query, query, peek_cur)
        attn_output, _ = self.self_attention(query, query, value, mask)
        attn_output = self.layer_norm1(query + attn_output)
        attn_output = self.dropout(attn_output)
        ffn_output = self.ffn(attn_output)
        encoder_out = self.layer_norm2(query + ffn_output)
        encoder_out = self.dropout(encoder_out)
        return encoder_out