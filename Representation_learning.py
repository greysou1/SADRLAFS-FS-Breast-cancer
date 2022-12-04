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
        self.Econv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.Tanh(),                      # activation
            #nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1, count_include_pad=False)
        )
        self.Econv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 3, 1, 1),     # output shape (32, 14, 14)
            nn.Tanh(),                      # activation
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1, count_include_pad=False)
        )
        self.Econv3 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(32,16, 3, 1, 1),     # output shape (32, 14, 14)
            nn.Tanh(),                      # activation
        )
        self.Espp_layer = SPPLayer(spp_level, 'avg_pool')
        # self.ELinear_layer = nn.Sequential(
        #     nn.Linear(1760, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 64),
        #     nn.Tanh(),
        # )
        # self.DLinear_layer = nn.Sequential(
        #     nn.Linear(64, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 1760),
        #     nn.Tanh(),
        # )
        self.Dconv1 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 3, 1, 1),     # output shape (32, 14, 14)
            nn.Tanh(),                      # activation
        )
        self.Dconv2 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.Tanh(),
        )
        self.Dconv3 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 1, 3, 1, 1),     # output shape (32, 14, 14)
            nn.Tanh(),                      # activation
        )


    def forward(self, x):
        num2, c2, h2, w2 = x.size()
        encoded = self.Econv1(x)
        num4, c4, h4, w4 = encoded.size()
        encoded = self.Econv2(encoded)
        num1, c1, h1, w1 = encoded.size()
        encoded = self.Econv3(encoded)
        num5, c5, h5, w5 = encoded.size()
        encoded = self.Espp_layer(encoded)
        # encoded = self.ELinear_layer(encoded)
        # decoded = self.DLinear_layer(encoded)
        index=0
        pooling = torch.zeros((num5, c5, h5, w5))
        for i in range(self.spp_level):
            ttensor = encoded[:, (index):(index+c5*(i+1)*(i+1))]
            ttensor = ttensor.view(num5, c5, (i+1), (i+1))
            index = index+c5*(i+1)
            ttensor = F.interpolate(ttensor, size=(h5, w5), mode='bilinear')
            pooling =torch.add(pooling, ttensor)
        decoded = pooling/self.spp_level
        decoded = self.Dconv1(decoded)
        decoded = F.interpolate(decoded, size=(h4, w4), mode='bilinear')
        decoded = self.Dconv2(decoded)
        decoded = F.interpolate(decoded, size=(h2, w2), mode='bilinear')
        decoded = self.Dconv3(decoded)
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

        # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
    return encoded