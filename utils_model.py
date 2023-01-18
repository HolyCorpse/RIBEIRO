import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import math

# RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO RIBEIRO

def need_zeros(tensor, desired, dimension):
    base = desired - tensor.shape[dimension]
    if base%2 == 1:
        pad = nn.ConstantPad1d((base//2, base//2 + 1), 0)
    elif base%2 == 0:
        pad = nn.ConstantPad1d((base//2, base//2), 0)

    # print(f'pad need zeros = {pad}')
    return pad(tensor)

def need_padding_same_stride_2(tensor, kernel_size):
    pad_value = (2 + kernel_size) // 2
    if pad_value % 2 == 0:
        pad = nn.ConstantPad1d((pad_value//2, pad_value//2), 0)
    elif pad_value % 2 == 1:
        pad = nn.ConstantPad1d((pad_value//2, pad_value//2 + 1), 0)

    # print(f'pad need padding same = {pad}')
    return pad(tensor)

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        # nn.init.xavier_normal_(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight)

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, rate_drop=0):
        super().__init__()
        self.skip = nn.Sequential(nn.MaxPool1d(2, 2), nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding='same'))

        self.first_middle = nn.Sequential(nn.BatchNorm1d(in_channel, eps=1e-5, momentum=0.99), nn.ReLU(), nn.Dropout(rate_drop))

        self.second_middle = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=15, stride=1, padding='same'),
                                    nn.BatchNorm1d(out_channel, eps=1e-5, momentum=0.99), nn.ReLU(), nn.Dropout(rate_drop),
                                    nn.Conv1d(out_channel, out_channel, kernel_size=15, stride=2))


        # self.apply(_weights_init)

    def forward(self, x):
        skip = self.skip(x)
        out = self.first_middle(x)
        out = need_padding_same_stride_2(out, kernel_size=15)
        out = self.second_middle(out)
        if out.shape[2] != skip.shape[2]:
            out = need_zeros(out, skip.shape[2], 2)
        out = torch.add(out, skip)
        return out

class RIBEIRO(nn.Module):
    def __init__(self, block, channels, rate_drop):
        #Initial Layers
        super().__init__()
        self.initial_op = nn.Sequential(nn.Conv1d(12, channels[0], kernel_size=15, stride=1, padding='same'),
                                        nn.BatchNorm1d(channels[0], eps=1e-05, momentum=0.99), nn.ReLU())

        self.pre_block = nn.Sequential(nn.Conv1d(channels[0], channels[1], kernel_size=15, stride=1, padding='same'),
                                                            nn.BatchNorm1d(channels[1], eps=1e-5, momentum=0.99), nn.ReLU(), nn.Dropout(rate_drop),
                                                            nn.Conv1d(channels[1], channels[1], kernel_size=15, stride=2))
        self.pre_skip  = nn.Sequential(nn.MaxPool1d(2, 2), nn.Conv1d(
            channels[0], channels[1], kernel_size=1, stride=1, padding='same'))

        self.layer1 = block(channels[1], channels[2], rate_drop)
        self.layer2 = block(channels[2], channels[3], rate_drop)
        self.layer3 = block(channels[3], channels[4], rate_drop)

        self.start_end = nn.Sequential(nn.BatchNorm1d(channels[4], eps=1e-5, momentum=0.99), nn.ReLU(), nn.Dropout(rate_drop))
        self.end = nn.Sequential(nn.Flatten(), nn.Linear(19840, 5), nn.Sigmoid())

        self.apply(_weights_init)
        # End of End Layer

    def forward(self, x):
        out = self.initial_op(x)
        skip = self.pre_skip(out)
        out = need_padding_same_stride_2(out, kernel_size=15)
        out = self.pre_block(out)
        if out.shape[2] != skip.shape[2]:
            out = need_zeros(out, skip.shape[2], 2)
        out = torch.add(out, skip)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.start_end(out)
        out = self.end(out)
        return out

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx, :, :]
        current_label = self.labels[idx, :]
        return current_sample, current_label
