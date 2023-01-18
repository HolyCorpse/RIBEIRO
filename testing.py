from tkinter import N
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import modelo_light as nutils
import utils_model as mutils
import pytorch_lightning.callbacks as callback
import utils

from sklearn.metrics import precision_score, recall_score, classification_report

batch_size = 107

data = np.load("data.npz")

X_train = data[f"X_train"]
X_test = data[f"X_test"]
y_train = data[f"y_train"].astype(np.float32)
y_test = data[f"y_test"].astype(np.float32)

data_lenght = X_train.shape[2]
data_channels = X_train.shape[1]
data_dimensions = (data_channels, data_lenght)

train = mutils.CustomDataset(X_train, y_train)
test = mutils.CustomDataset(X_test, y_test)

train_dl = DataLoader(train, batch_size=batch_size)
test_dl = DataLoader(test, batch_size=batch_size)

dirpath = f'saved_models/saved_model:2023-01-09T15:22:37.690770'
# dirpath = f'saved_models/saved_model:2023-01-06T13:38:29.226212'
filename = f'/resnet_saved_model.ckpt'

# modelIC = mutils.ModeloAndrew(mutils.BasicBlock, option='A', initial_filters=64, num_blocks=[
#                               1, 1, 1, 1], strides=[2, 2, 2, 2], dimensions=data_dimensions)

model = nutils.LitModel.load_from_checkpoint(dirpath+filename)
model.eval()
sigmoid_de_cria = torch.nn.Sequential(torch.nn.Sigmoid())
# y_hat = sigmoid_de_cria(model(torch.tensor(X_test)))
y_hat = model(torch.tensor(X_test))
# y_hat = model(torch.tensor(X_train[0:3000, :, :]))
y_hat = y_hat.detach().numpy()

for i in range(y_hat.shape[0]):
    for j in range(y_hat.shape[1]):
        if y_hat[i, j] > 0.5:
            y_hat[i, j] = 1
        else:
            y_hat[i, j] = 0

diag_names = ['NORM', 'STTC', 'CD', 'MI', 'HYP']

report_test = classification_report(
    y_true=y_test, y_pred=y_hat, output_dict=False, target_names=diag_names)
# report_test = classification_report(
#     y_true=y_train[0:3000,:], y_pred=y_hat, output_dict=False, target_names=diag_names)

print(report_test)

cm = utils.plot_confusion_matrix(y_test, y_hat, "resnet_de_cria", diag_names)
# cm = utils.plot_confusion_matrix(y_train[0:3000,:], y_hat, "resnet_de_cria", diag_names)

print()
