from dataset.sequence_to_price import SeqToPriceDataset


class SeqToClassDataset(SeqToPriceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        seq_x, seq_y = super().__getitem__(i)

        return seq_x, (seq_y > 0).long().squeeze(-1)
