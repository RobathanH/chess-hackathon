import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, nhead, nhid, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, nhid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(nhid, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Model(nn.Module):
    """Transformer Model"""

    def __init__(self, ntoken, embed_dim, nhead, nhid, nlayers, batch_first, dropout=0.5):
        super(Model, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        self.input_emb = nn.Embedding(ntoken, embed_dim)
        self.embed_dim = embed_dim

        self.attention_layers = nn.ModuleList([AttentionLayer(embed_dim, nhead, nhid, dropout) for _ in range(nlayers)])
        self.decoder = nn.Linear(64 * embed_dim, 1, bias=False)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, inputs): # (candidate, square)
        inputs = inputs.flatten(1)
        inputs = self.input_emb(inputs) * math.sqrt(self.embed_dim) # (candidate, square, embed)
        inputs = self.pos_encoder(inputs) # (candidate, square, embed)
        
        for attention_layer in self.attention_layers:
            inputs = attention_layer(inputs)
        
        latents = torch.flatten(inputs, 1) # (candidate, square_embed)
        scores = self.decoder(latents).flatten() # (candidate)
        return scores
