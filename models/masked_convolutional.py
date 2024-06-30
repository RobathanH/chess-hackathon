import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedConv2d(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        kernel_masks = [
            # Pawn
            np.pad(np.array([
                [1, 0, 1],
                [0, 1, 0],
                [0, 0, 0]
            ]).astype(bool), 6),
            # Knight
            np.pad(np.array([
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0]
            ]).astype(bool), 5),
            # Bishop
            np.diag(np.ones(15, dtype=bool)) | np.flip(np.diag(np.ones(15, dtype=bool)), axis=0),
            # Rook
            np.array([
                [
                    (i == 7 or j == 7)
                    for j in range(15)
                ]
                for i in range(15)
            ]),
            # Queen
            np.array([
                [
                    (i == 7 or j == 7)
                    for j in range(15)
                ]
                for i in range(15)
            ]) | np.diag(np.ones(15, dtype=bool)) | np.flip(np.diag(np.ones(15, dtype=bool)), axis=0),
            # King
            np.pad(np.ones((3, 3), dtype=bool), 6)
        ]
        for k in kernel_masks:
            print("\n")
            print(k.astype(int))

        # Repeat kernel masks for each channel out
        kernel_masks = [
            kernel_masks[i % len(kernel_masks)]
            for i in range(channels_out)
        ]
        kernel_masks = torch.from_numpy(np.stack(kernel_masks)).unsqueeze(1)
        self.register_buffer("kernel_masks", kernel_masks)

        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=15, padding=7, stride=1, bias=False)
        print(self.kernel_masks.shape)
        print(self.conv.weight.shape)

    def mask_conv_weights(self):
        with torch.no_grad():
            self.conv.weight *= self.kernel_masks
    
    def forward(self, x):
        self.mask_conv_weights()
        return self.conv(x)

class Residual(nn.Module):
    """The Residual block of ResNet models."""
    def __init__(self, outer_channels, inner_channels, use_1x1conv=False):
        super().__init__()
        self.conv1 = MaskedConv2d(outer_channels, inner_channels)
        self.conv2 = MaskedConv2d(inner_channels, outer_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(outer_channels, outer_channels, kernel_size=1, stride=1)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(inner_channels)
        self.bn2 = nn.BatchNorm2d(outer_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class Model(nn.Module):
    """Convolutional Model"""

    def __init__(self, ntoken, embed_dim, nlayers, dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ntoken = ntoken
        self.embed_dim = embed_dim

        self.input_emb = nn.Embedding(self.ntoken, self.embed_dim)
        self.convnet = nn.Sequential(*[Residual(self.embed_dim, 5 * self.embed_dim, use_1x1conv=True) for _ in range(nlayers)])
        self.accumulator = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=8, padding=0, stride=1)
        self.decoder = nn.Linear(self.embed_dim, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, inputs): # (N, 8, 8)
        # print(inputs.shape)
        inputs = self.input_emb(inputs) # (N, 8, 8, D) - this is nice
        # print(inputs.shape)
        inputs = torch.permute(inputs, (0, 3, 1, 2))
        # print(inputs.shape)
        inputs = self.convnet(inputs)
        # print(inputs.shape)
        inputs = F.relu(self.accumulator(inputs).squeeze())
        # print(inputs.shape)
        scores = self.decoder(inputs).flatten()
        # print(scores.shape)
        return scores