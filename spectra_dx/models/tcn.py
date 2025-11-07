from __future__ import annotations
import torch.nn as nn

class CausalTCN(nn.Module):
    def __init__(self, in_ch: int, hid: int = 128, levels: int = 3, k: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_ch
        for l in range(levels):
            dil = 2**l
            pad = (k-1)*dil
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, hid, kernel_size=k, dilation=dil, padding=pad),
                nn.GELU(),
            ))
            ch = hid

    def forward(self, x):
        y = x.transpose(1, 2)
        L = y.size(-1)
        for b in self.blocks:
            y = b(y); y = y[..., :L]
        return y.transpose(1, 2)
