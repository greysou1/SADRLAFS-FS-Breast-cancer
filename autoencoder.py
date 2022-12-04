import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Autoencoder1(nn.Module):
    def __init__(self, spp_level=4):
        super(Autoencoder1, self).__init__()
        self.ELinear_layer = nn.Sequential(
            nn.Linear(1760, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.DLinear_layer = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 1760),
            nn.Tanh(),
        )

    def forward(self, x):

        encoded = self.ELinear_layer(encoded)
        decoded = self.DLinear_layer(encoded)
        return x, encoded, decoded


class Autoencoder2(nn.Module):
    def __init__(self, spp_level=4):
        super(Autoencoder2, self).__init__()
        self.ELinear_layer = nn.Sequential(
            nn.Linear(1760, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.DLinear_layer = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 1760),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.ELinear_layer(encoded)
        decoded = self.DLinear_layer(encoded)
        return x, encoded, decoded

def representation_training(train_data):

    # Hyper Parameters
    EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
    SLR = 0.005  # learning rate
    spp_cnn_autoencoder = Autoencoder1()

    optimizer = torch.optim.Adam(spp_cnn_autoencoder.parameters(), lr=SLR)
    loss_func = nn.MSELoss()

    for epoch in range(EPOCH):

        b_y, encoded, decoded = spp_cnn_autoencoder(train_data)
        loss = loss_func(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
    return encoded