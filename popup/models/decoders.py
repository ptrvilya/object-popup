"""
Decoders for PointNet based models.
"""
import torch.nn as nn


class DecoderPointNet2(nn.Module):
    def __init__(self, in_dim, out_dim=3, hidden_dim=256, with_bn=True, **kwargs):
        super(DecoderPointNet2, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False if with_bn else True),
            nn.BatchNorm1d(hidden_dim) if with_bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False if with_bn else True),
            nn.BatchNorm1d(hidden_dim // 2) if with_bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False if with_bn else True),
            nn.BatchNorm1d(hidden_dim // 4) if with_bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, out_dim),
        )

    def forward(self, features):
        return self.decoder(features)


class DecoderPointNet2Seg(nn.Module):
    def __init__(self, in_dim, out_dim=3, hidden_dim=256, with_bn=True, **kwargs):
        super(DecoderPointNet2Seg, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False if with_bn else True),
            nn.BatchNorm1d(hidden_dim) if with_bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False if with_bn else True),
            nn.BatchNorm1d(hidden_dim // 2) if with_bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False if with_bn else True),
            nn.BatchNorm1d(hidden_dim // 4) if with_bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, out_dim),
        )

    def forward(self, features):
        return self.decoder(features)
