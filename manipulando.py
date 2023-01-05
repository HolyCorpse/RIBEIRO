from tkinter import N
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import modelo_light as nutils
import utils_model as mutils
import pytorch_lightning.callbacks as callback


from pytorch_lightning.loggers import CSVLogger

from datetime import datetime

torch.cuda.empty_cache()

batch_size = 107

data = np.load("data.npz")

X_train = data[f"X_train"]
X_test = data[f"X_test"]
X_val = data[f"X_val"]
y_train = data[f"y_train"]
y_test = data[f"y_test"]
y_val = data[f"y_val"]

data_lenght = X_train.shape[2]
data_channels = X_train.shape[1]
data_dimensions = (data_channels, data_lenght)

num_classes = y_train.shape[1]

train = mutils.CustomDataset(X_train, y_train)
test = mutils.CustomDataset(X_test, y_test)
val = mutils.CustomDataset(X_val, y_val)

train_dl = DataLoader(train, batch_size=batch_size)
test_dl = DataLoader(test, batch_size=batch_size)
val_dl = DataLoader(val, batch_size=batch_size)


modelIC = nutils.LitModel(mutils.BasicBlock, channels=[64,128,192,256,320])
trainer = pl.Trainer(fast_dev_run=True, max_epochs=100,
                     accelerator='gpu')

trainer.fit(modelIC, train_dataloaders=train_dl, val_dataloaders=val_dl)

print(f"\n\n\n\nFim do código\n\n\n\n")
