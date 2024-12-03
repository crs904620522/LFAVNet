import torch
import torch.nn as nn
import yaml
import os
from lf2disp.LFAVNet.models.net import Net
class LFAVNet(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        self.device = device
        input_dim = cfg['LFAVNet']['input_dim']
        n_views = cfg['data']['views']
        self.net = Net(input_dim=input_dim, n_views=n_views, device=device).to(self.device)

    def forward(self, input, guide_index):
        B1, B2, H, W, _, M, M = input.shape
        input = input.reshape(B1 * B2, H, W, 1, M, M)
        depthmap = self.net(input, guide_index)
        B, H, W, C = depthmap.shape
        out = {'pred': depthmap.reshape(B1, B2, H, W, -1)}
        return out

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model