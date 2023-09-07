import torch
from torch import nn

from transformer.positional_encoding import PositionalEncoding
from transformer.encoder import Encoder

class Transformer(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        d_model,
        nhead,
        d_hid,
        num_layers,
        dropout,
    ):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model, batch_first=True)

        self.encoder = Encoder(
            num_inputs, d_model, nhead, d_hid, num_layers, dropout
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, num_outputs, bias=True)
        )

    def init_weights(self):
        init_range = 0.1

        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        memory = self.encoder(src)
        memory = torch.swapaxes(memory, 1, 2) # (batch_size, seq_len, d_model) -> (batch_size, d_model, seq_len)
        memory = nn.AvgPool1d(memory.shape[-1])(memory) # (batch_size, d_model, seq_len) ->  (batch_size, d_model, 1)
        memory = memory.squeeze(-1) # (batch_size, d_model, 1) -> (batch_size, d_model)
        
        out = self.output_projection(memory) # (batch_size, d_model) -> (batch_size, num_outputs)

        return out
        