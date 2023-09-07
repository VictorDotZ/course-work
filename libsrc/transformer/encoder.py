from torch import nn

from transformer.positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, num_enc_inputs, d_model, nhead, d_hid, num_layers, dropout):
        super().__init__()

        self.input_projection = nn.Linear(num_enc_inputs, d_model, bias=True)

        self.pos_encoder = PositionalEncoding(d_model, batch_first=True)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1

        self.input_projection.bias.data.zero_()
        self.input_projection.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, mask=None):
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, mask=mask)

        return memory
