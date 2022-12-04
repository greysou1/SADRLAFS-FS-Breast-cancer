import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size() #num:样本数量 c:通道数 h:高 w:宽
        pooling_layers = []
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.floor(h / level), math.floor(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))
            zero_pad = torch.nn.ZeroPad2d((pooling[1], pooling[1], pooling[0], pooling[0]))
            x_new = zero_pad(x)
            h_new = 2*pooling[0] + h
            w_new = 2*pooling[1] + w
            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
            else:
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


class SPP_CNN_Autoencoder(nn.Module):
    def __init__(self, spp_level=4):
        super(SPP_CNN_Autoencoder, self).__init__()
        self.spp_level = spp_level
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.Tanh(),                      # activation
            #nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.AvgPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 3, 1, 1),     # output shape (32, 14, 14)
            nn.Tanh(),                      # activation
        )
        self.spp_layer = SPPLayer(spp_level, 'Avg_pool')
        self.encoder = nn.Sequential(
            nn.Linear(960, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 960),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.spp_layer(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return x, encoded, decoded


def representation_training(train_data):

    # Hyper Parameters
    EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
    SLR = 0.005  # learning rate
    spp_cnn_autoencoder = SPP_CNN_Autoencoder()

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