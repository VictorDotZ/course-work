from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout, batch_first=True):
        super().__init__()

        self.rnn = nn.LSTM(
            input_dim,
            hid_dim,
            n_layers,
            dropout=dropout,
            batch_first=batch_first,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        out, (_, _) = self.rnn(src)
        return out
