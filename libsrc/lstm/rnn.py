from torch import nn

from lstm.encoder import Encoder


class RNN(nn.Module):
    def __init__(
        self, input_dim, hid_dim, output_dim, n_layers, dropout, mode="regression"
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            hid_dim=hid_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.output_projection = nn.Linear(hid_dim * 2, output_dim, bias=True)

        if mode == "classification":
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, src):
        out = self.encoder(src)
        out = self.output_projection(out)

        if hasattr(self, "softmax"):
            out = self.softmax(out)

        return out[:, -1, :]
