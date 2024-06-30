import math
import torch
import torch.nn as nn

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

        self.norm1 = nn.BatchNorm1d(d_model*2)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        x = self.norm1(x)

        return x

class Model(nn.Transformer):
    """Transformer Model"""

    def __init__(self, ntoken, embed_dim, nhead, nhid, nlayers, batch_first, dropout=0.5):
        super(Model, self).__init__(d_model=embed_dim, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, batch_first=batch_first)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        self.input_emb = nn.Embedding(ntoken, embed_dim)
        self.embed_dim = embed_dim
        self.decoder = nn.Linear(64 * embed_dim, 1, bias=False)

        self.embed_norm = nn.BatchNorm1d(embed_dim)

        self.linear = nn.Linear(64,1,bias=True,dtype=torch.float32)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, inputs): # (candidate, square)
        # print(inputs.shape)
        print("INPUT RAW SHAPE:")
        print(inputs.shape)
        inputs = inputs.flatten(1)
        # print(inputs.shape)
        #inputs = self.input_emb(inputs) * math.sqrt(self.embed_dim) # (candidate, square, embed)
        print("INPUT RESHAPE:")
        print(inputs.shape)
        inputs = inputs.to(torch.float32)
        # norm
        #inputs = self.embed_norm(inputs)
        #print(inputs.shape)

        # print(inputs.shape)
        # inputs = self.pos_encoder(inputs) # (candidate, square, embed)
        # # print(inputs.shape)
        # encoded = self.encoder(inputs) # (candidate, square, embed)
        # # norm
        # encoded = self.embed_norm(encoded)
        # # print(encoded.shape)
        # latents = torch.flatten(encoded, 1) # (candidate, square_embed)
        # # print(latents.shape)

        scores = self.linear(inputs) # (candidate)
        print("SCORES SHAPE")
        print(scores.shape)
        # print(scores.shape)
        return scores