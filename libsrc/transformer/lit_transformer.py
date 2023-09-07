import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics import F1Score
from torchmetrics import Precision
from torchmetrics import Recall

from transformer.transformer import Transformer


class LitTransformer(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model,
        nhead,
        hid_dim,
        n_layers,
        dropout,
        mode="classification",
    ):
        super().__init__()
        self.model = Transformer(
            input_dim, output_dim, d_model, nhead, hid_dim, n_layers, dropout
        )

        if mode == "classification":
            self.softmax = torch.nn.Softmax(dim=-1)
            self.val_acc = Accuracy(task="multiclass", num_classes=2)
            self.val_f1 = F1Score(task="multiclass", num_classes=2)
            self.val_prec = Precision(task="multiclass", num_classes=2)
            self.val_rec = Recall(task="multiclass", num_classes=2)
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif mode == "regression":
            self.loss_function = torch.nn.MSELoss()

    def forward(self, src):
        out = self.model.forward(src)

        if hasattr(self, "softmax"):
            out = self.softmax(out)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        src, y = train_batch
        out = self.forward(src)
        loss = self.loss_function(out, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        # pylint: disable=not-callable
        src, y = val_batch
        out = self.forward(src)
        loss = self.loss_function(out, y)
        if hasattr(self, "val_acc"):
            self.val_acc(out, y)
        if hasattr(self, "val_f1"):
            self.val_f1(out, y)
        if hasattr(self, "val_prec"):
            self.val_prec(out, y)
        if hasattr(self, "val_rec"):
            self.val_rec(out, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_epoch_end(self, *args):
        if hasattr(self, "val_acc"):
            self.log("val_acc_epoch", self.val_acc.compute())
            self.val_acc.reset()
        if hasattr(self, "val_f1"):
            self.log("val_f1_epoch", self.val_f1.compute())
            self.val_f1.reset()
        if hasattr(self, "val_prec"):
            self.log("val_prec_epoch", self.val_prec.compute())
            self.val_prec.reset()
        if hasattr(self, "val_rec"):
            self.log("val_rec_epoch", self.val_rec.compute())
            self.val_rec.reset()
