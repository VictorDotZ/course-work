import torch
from torch.utils.data import Dataset


class SeqToPriceDataset(Dataset):
    def __init__(
        self,
        dataframe,
        features=None,
        targets=None,
        sequence_length=96,
        forecast_length=24,
    ):
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length

        self.features_ids = torch.tensor(dataframe.columns.get_indexer(features))
        self.targets_ids = torch.tensor(dataframe.columns.get_indexer(targets))
        self.data = torch.tensor(dataframe.values).float()

    def __len__(self):
        return self.data.shape[0] - self.sequence_length - self.forecast_length + 1

    def __getitem__(self, i):
        src_start = i
        src_end = src_start + self.sequence_length
        y_start = src_end - 1
        y_end = y_start + 1 + self.forecast_length

        seq_x = self.data[src_start:src_end]
        seq_y = self.data[y_end - 1]

        return seq_x[:, self.features_ids], seq_y[self.targets_ids]
