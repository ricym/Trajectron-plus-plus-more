import numpy as np

from graphtranformers import *
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_t_len=200, max_a_len=200, concat=False, use_agent_enc=False,
                 agent_enc_learn=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        self.use_agent_enc = use_agent_enc
        if concat:
            self.fc = nn.Linear((3 if use_agent_enc else 2) * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        if use_agent_enc:
            if agent_enc_learn:
                self.ae = nn.Parameter(torch.randn(max_a_len, 1, d_model) * 0.1)
            else:
                ae = self.build_pos_enc(max_a_len)
                self.register_buffer('ae', ae)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def build_agent_enc(self, max_len):
        ae = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtyae=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        ae[:, 0::2] = torch.sin(position * div_term)
        ae[:, 1::2] = torch.cos(position * div_term)
        ae = ae.unsqueeze(0).transpose(0, 1)
        return ae

    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe.repeat_interleave(num_a, dim=0)
        return pe  # [N*T 1 model_dim]

    def get_agent_enc(self, num_t, num_a, a_offset, agent_enc_shuffle):
        if agent_enc_shuffle is None:
            ae = self.ae[a_offset: num_a + a_offset, :]
        else:
            ae = self.ae[agent_enc_shuffle]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, agent_enc_shuffle=None, t_offset=0, a_offset=0):
        num_t = x.shape[0] // num_a  # x [N*T 1 model_dim]
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset)
        if self.use_agent_enc:
            agent_enc = self.get_agent_enc(num_t, num_a, a_offset, agent_enc_shuffle)
        if self.concat:
            feat = [x, pos_enc.repeat(1, x.size(1), 1)]
            if self.use_agent_enc:
                feat.append(agent_enc.repeat(1, x.size(1), 1))
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
            if self.use_agent_enc:
                x += agent_enc
        return self.dropout(x)


class GraphHistoryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size,
                obs_len=8, dropout=0, bias=False, pooling=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.dropout = dropout
        self.bias = bias
        self.pooling = pooling

        self.input_embeddings = nn.Linear(self.input_size, self.hidden_size)
        self.pos_encoder = PositionalEncoding(self.model_dim, self.dropout)

    def forward(self, traj_in, agent_mask=None, agent_enc_shuffle=None):
        # traj: [bs, traj_num, timesteps, fra=6(x, y, vx, vy, ax, ay)]
        agent_num = traj_in.shape[1]

        tin_pos = self.
        ind_p = ind

        if self.pooling:
            agent_history = torch.mean(history, dim=0)
        else:
            agent_history = torch.max(history, dim=0)

        return history_enc, agent_history

class GraphFutureEncoder(nn.Module):
    def __init__(self, input_size, hidden_size,
                pred_len=12, dropout=0, bias=False):
        super().__init__()

        self.pred_len = pred_len

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.dropout = dropout
        self.bias = bias

        self.pos_encoder = PositionalEncoding(self.model_dim, self.dropout)

    def forward(self, traj_in, agent_mask=None, agent_enc_shuffle=None):
        # traj: [bs, traj_num, timesteps, fra=6(x, y, vx, vy, ax, ay)]
        agent_num = traj_in.shape[1]

        tin_pos = self.
        ind_p = ind

        return graph, ind_p, ind_n