"""
drmodules.py  -  Extra layers for the LA-DRNet-Lite detector
=============================================================

  SimAM : parameter-free 3-D attention (Yang et al., ICML 2021). Adds 0 parameters.
  CBAM  : Convolutional Block Attention Module (Woo et al., ECCV 2018), original form:
          channel attention (avg + max pooling -> shared MLP, reduction 16)
          followed by spatial attention (avg + max over channels -> 7x7 conv).

Keep this file in F:/drthesis/code/ next to the scripts. Trained models that use these
layers can only be loaded by a script that has done  `import drmodules`  first.
"""
import torch
import torch.nn as nn


class SimAM(nn.Module):
    """Weights every activation by how much it stands out from the rest of its channel."""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        xf = x.float()                                     # float32: sums over 256x256 maps overflow fp16
        d = (xf - xf.mean(dim=(2, 3), keepdim=True)).pow(2)
        energy = d / (4 * (d.sum(dim=(2, 3), keepdim=True) / max(n, 1) + self.e_lambda)) + 0.5
        return x * torch.sigmoid(energy).to(x.dtype)


class CBAM(nn.Module):
    """CBAM as in the original paper. c1 = number of input channels."""

    def __init__(self, c1, reduction=16, kernel_size=7):
        super().__init__()
        hidden = max(c1 // reduction, 1)
        self.mlp = nn.Sequential(nn.Conv2d(c1, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, c1, 1))
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
                                     nn.BatchNorm2d(1))

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


def register():
    """Make the layers usable by name inside Ultralytics model YAML files."""
    import ultralytics.nn.tasks as tasks
    tasks.SimAM = SimAM
    tasks.CBAM = CBAM


register()
