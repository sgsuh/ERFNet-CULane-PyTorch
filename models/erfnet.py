# ERFNET full network definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

"""
Modify: 2021.09.07
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

from torch.autograd import Variable

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride = 2, padding = 1, bias = True)
        self.pool = nn.MaxPool2d(2, stride = 2)
        self.bn = nn.BatchNorm2d(noutput, eps = 1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)

class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride = 1, padding = (1, 0), bias = True)
        self.bn0_1 = nn.BatchNorm2d(chann, eps = 1e-03)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride = 1, padding = (0, 1), bias = True)
        self.bn1 = nn.BatchNorm2d(chann, eps = 1e-03)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride = 1, padding = (1 * dilated, 0), bias = True, dilation = (dilated, 1))
        self.bn0_2 = nn.BatchNorm2d(chann, eps = 1e-03)
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride = 1, padding = (0, 1 * dilated), bias = True, dilation = (1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps = 1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.bn0_1(output)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = self.bn0_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)

class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        
        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):
            self.layers.append(non_bottleneck_1d(64, 0.1, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))

        # Only for Encoder Mode
        self.output_conv = nn.Conv2d(128, num_classes, 1,  stride = 1, padding = 0, bias = True)

    def forward(self, input, predict = False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output

class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride = 2, padding = 1, output_padding = 1, bias = True)
        self.bn = nn.BatchNorm2d(noutput, eps = 1e-3, track_running_stats = True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)

        return F.relu(output)

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride = 2, padding = 0, bias = True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

class Lane_exist(nn.Module):
    def __init__(self, num_output):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(128, 32, (3, 3), stride = 1, padding = (4, 4), bias = False, dilation = (4, 4)))
        self.layers.append(nn.BatchNorm2d(32, eps = 1e-03))

        self.layers_final = nn.ModuleList()

        self.layers_final.append(nn.Dropout2d(0.1))
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride = 1, padding = (0, 0), bias = True))

        self.maxpool = nn.MaxPool2d(2, stride = 2)
        self.linear1 = nn.Linear(5100, 128)     # 3965
        self.linear2 = nn.Linear(128, 4)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = F.relu(output)

        for layer in self.layers_final:
            output = layer(output)

        output = F.softmax(output, dim=1)
        output = self.maxpool(output)
        
        output = output.view(-1, 5100)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = torch.sigmoid(output)

        return output

class ERFNet(nn.Module):
    def __init__(self, num_classes, encoder = None):        # Use Encoder to Pass Pretrained Encoder
        super().__init__()

        if encoder == None:
            self.encoder = Encoder(num_classes) 
        else:
            self.encoder = encoder

        self.decoder = Decoder(num_classes)
        
        self.input_mean = [103.939, 116.779, 123.68]        # [0, 0, 0]
        self.input_std = [1, 1, 1]

    def train(self, mode = True):
        # Override the Default train() to Freeze the BN Parameters
        super(ERFNet, self).train(mode)

    def get_optim_policies(self):
        base_weight = []
        base_bias = []
        base_bn = []

        addtional_weight = []
        addtional_bias = []
        addtional_bn = []

        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                ps = list(m.parameters())

                base_weight.append(ps[0])

                if len(ps) == 2:
                    base_bias.append(ps[1])
            elif isinstance(m, nn.BatchNorm2d):
                base_bn.extend(list(m.parameters()))

        return [{'params': addtional_weight, 'lr_mult': 10, 'decay_mult': 1, 'name': 'addtional_weight'},
                {'params': addtional_bias, 'lr_mult': 20, 'decay_mult': 1, 'name': 'addtional_bias'},
                {'params': addtional_bn, 'lr_mult': 10, 'decay_mult': 0, 'name': 'addtional_bn_scale/shift'},
                {'params': base_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': 'base_weight'},
                {'params': base_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': 'base_bias'},
                {'params': base_bn, 'lr_mult': 1, 'decay_mult': 0, 'name': 'base_bn_scale/shift'}]

    def forward(self, input):
        x = F.pad(input, [0, 0, 0, 4, 0, 0], mode = 'constant', value = 0)

        encoder = self.encoder(x)
        decoder = self.decoder.forward(encoder)
        
        return decoder[:, :, :-4, :], None

class LaneNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = Encoder(num_classes[0])
        self.est_decoder = Decoder(num_classes[0])
        self.seg_decoder = Decoder(num_classes[1])

        self.input_mean = Variable(torch.Tensor((103.939, 116.779, 123.68)), requires_grad = False).cuda()
        self.input_std = Variable(torch.Tensor((1.0, 1.0, 1.0)), requires_grad = False).cuda()

    def forward(self, input):
        input = (input - self.input_mean) / self.input_std
        input = input.permute(0, 3, 1, 2)
        encoder = self.encoder(input)

        est_decoder = self.est_decoder(encoder)
        seg_decoder = self.seg_decoder(encoder)

        est = F.softmax(est_decoder, dim = 1).permute(0, 2, 3, 1)
        seg = F.softmax(seg_decoder, dim = 1).permute(0, 2, 3, 1)

        return est[:, :-2, :, :], seg[:, :-2, :, :]                        