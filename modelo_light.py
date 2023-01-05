import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import utils_model as mutils

class LitModel(pl.LightningModule):
    def __init__(self, block, channels, rate_drop):
        super().__init__()
        self.save_hyperparameters()
        self.model = mutils.RIBEIRO(
            block, channels, rate_drop)
        self.metric = torch.nn.MultiLabelSoftMarginLoss(
            weight=None, reduction='mean'
        )
        # self.metric = torch.nn.BCELoss()

    def forward(self, x):
        return self.model.forward(x)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #     return [optimizer], [lr_scheduler]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.metric(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.metric(y_hat, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        test_loss = self.metric(y_hat, y)
        self.log("test_loss", test_loss)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        pred = self(x)
        return pred

if __name__ == "__main__":
    import utils_model as mutils
    modelo = mutils.ModeloAndrew()
    modeloLight = LitModel(modelo)
    print(f"Fim do codigo")
