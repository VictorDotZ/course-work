import torch
import numpy as np


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, batch_first=False, max_len=5000):
        super().__init__()
        self.batch_first = batch_first
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)  # не параметр модели ни в коем случае

    def forward(self, x):
        if self.batch_first:
            # кодироваться должны dt чтобы отличаться друг от друга, а не батчи
            x = x + self.pe[: x.size(1)].permute(1, 0, 2).to(x)
        else:
            x = x + self.pe[: x.size(0)].to(x)
        return x
