import math
import torch
import torch.nn as nn

class RelativePosition1d(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k, device):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).to(device)
        embeddings = self.embeddings_table[final_mat].to(device)

        return embeddings

class RelativePosition2d(nn.Module):
    def __init__(self, embed_dim, input_shape):
        super().__init__()

        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.input_shape = input_shape

        self.relative_position_x = RelativePosition1d(embed_dim // 2, input_shape[0] - 1)
        self.relative_position_y = RelativePosition1d(embed_dim // 2, input_shape[1] - 1)

    def forward(self, device):
        # Assumes self attention (k,q same shape) and fixed flattened length
        x_pos = self.relative_position_x(self.input_shape[0], self.input_shape[0], device)[:, None, :, None, :]\
            .expand(self.input_shape[0], self.input_shape[1], self.input_shape[0], self.input_shape[1], self.embed_dim // 2)
        y_pos = self.relative_position_y(self.input_shape[1], self.input_shape[1], device)[None, :, None, :, :]\
            .expand(self.input_shape[0], self.input_shape[1], self.input_shape[0], self.input_shape[1], self.embed_dim // 2)
        return torch.concat([x_pos, y_pos], dim=-1)\
            .reshape(self.input_shape[0] * self.input_shape[1], self.input_shape[0] * self.input_shape[1], self.embed_dim)


class RelMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, input_shape):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.relative_position_k = RelativePosition2d(self.head_dim, input_shape)
        self.relative_position_v = RelativePosition2d(self.head_dim, input_shape)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))#.to(device)

        self.batch_first = True
        self.in_proj_bias = None
        
    def forward(self, query, key, value, attn_mask = None, 
                key_padding_mask=None,
                need_weights=False, is_causal=False):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(query.device)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale.to(query.device)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(query.device)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x

class Model(nn.Transformer):
    """Transformer Model"""

    def __init__(self, ntoken, embed_dim, nhead, nhid, nlayers, batch_first, dropout=0.5):
        super(Model, self).__init__(d_model=embed_dim, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, batch_first=batch_first)
        self.model_type = 'Transformer'
        self.src_mask = None

        self.input_emb = nn.Embedding(ntoken, embed_dim)
        self.embed_dim = embed_dim

        # Set relative attention for encoder layers
        for layer in self.encoder.layers:
            layer.self_attn = RelMultiHeadAttentionLayer(embed_dim, nhead, dropout, (8, 8))

        self.decoder = None

        self.agg_attn_key = nn.Parameter(torch.zeros((1, 1, embed_dim)))

        self.score_head = nn.Linear(embed_dim, 1, bias=True)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.uniform_(self.agg_attn_key, -initrange, initrange)

    def forward(self, inputs): # (candidate, square)
        # print(inputs.shape)
        inputs = inputs.flatten(1)
        # print(inputs.shape)
        inputs = self.input_emb(inputs) * math.sqrt(self.embed_dim) # (candidate, square, embed)
        # print(inputs.shape)
        encoded = self.encoder(inputs) # (candidate, square, embed)
        # print(encoded.shape)

        # Attention-based aggregation
        attn_weights = torch.softmax((self.agg_attn_key * encoded).sum(2), dim=1)
        latents = (attn_weights[:, :, None] * encoded).sum(1)
        # print(latents.shape)
        scores = self.score_head(latents).flatten() # (candidate)
        # print(scores.shape)
        return scores