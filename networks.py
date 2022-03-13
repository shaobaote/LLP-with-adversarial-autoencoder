import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torchvision import datasets,transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torchvision,torch


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, input):
        if self.training:
            noise = Variable(input.data.new(input.size()).normal_(std=self.sigma))
            return input + noise
        else:
            return input

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Encoder_grey(nn.Module):
    def __init__(self,y_dim,z_dim,method='llp_aae'):
        super(Encoder_grey,self).__init__()
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.method=method
        
        #add the gaussian noise on the image
        self.gaosi = nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.15)) 

        self.model=nn.Sequential(
            nn.Linear(784,1000),nn.LeakyReLU(),
            nn.Linear(1000,1000),nn.LeakyReLU(),
        )
        self.cat=nn.Sequential(
            nn.Linear(1000,y_dim),nn.Softmax(dim=1)
        )
        self.gauss=nn.Linear(1000,z_dim)
        
        self.apply(weights_init)
        
    def forward(self,x):
        x=x.view(x.shape[0],-1)
        x=self.model(x)
        y_cat = self.cat(x)
        z_gauss = self.gauss(x)
        
        if self.method=='dllp':return y_cat
        elif self.method=='llp_aae' or self.method=='llp_gan' or self.method=='ss_aae':return y_cat,z_gauss
        elif self.method=='aae':return z_gauss

class Decoder_grey(nn.Module):
    def __init__(self,y_dim,z_dim, method, output_dim=1, input_size=28):
        super(Decoder_grey,self).__init__()
        if method=='aae':
            self.input_dim=z_dim
        else:
            self.input_dim=y_dim+z_dim
        self.z_dim=z_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 1000), nn.LeakyReLU(),
            nn.Linear(1000, 784), nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, X):
        out = self.net(X)
        out=out.view(out.shape[0],-1,28,28)
        
        return out
    
class Encoder_RGB(nn.Module):
    def __init__(self, y_dim,z_dim,method):
        super(Encoder_RGB, self).__init__()

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.method=method
        #self.dataset_name=dataset_name

        #add the gaussian noise on the image
        #self.gaosi = nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.15)) 

        self.core_net1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2),
            nn.Dropout(0.5)) 

        self.core_net2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2),
            nn.Dropout(0.5)) 


        self.core_net3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False), nn.LeakyReLU(0.2),
            #nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.LeakyReLU(0.2),
            )

        self.avgpool=nn.AvgPool2d(6)


        self.out_net1 = nn.Sequential(
                #nn.Linear(64, self.y_dim),nn.Softmax(dim=1)
                nn.Linear(128, self.y_dim),nn.Softmax(dim=1)
            )
        #self.out_net2 = nn.Linear(64, self.z_dim)
        self.out_net2 = nn.Linear(128, self.z_dim)
        
        self.apply(weights_init)

    def forward(self, x):
        #x = self.gaosi(x)
        x = self.core_net1(x)
        x = self.core_net2(x)
        x = self.core_net3(x)
        x_avg = self.avgpool(x)
        x = x_avg.view(x_avg.size(0), -1)
        if self.method=='dllp':return self.out_net1(x)
        elif self.method=='llp_aae' or self.method=='llp_gan'  or self.method=='ss_aae':return self.out_net1(x), self.out_net2(x)
        elif self.method=='aae':return self.out_net2(x)

class Decoder_RGB(nn.Module):
    def __init__(self, y_dim, z_dim, method, output_dim=3, input_size=32):
        super(Decoder_RGB, self).__init__()
        self.output_dim = output_dim
        self.input_size = input_size
        self.y_dim = y_dim
        self.z_dim = z_dim
        if method=='aae':
            self.input_dim=z_dim
        else:
            self.input_dim=y_dim+z_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 512 * (self.input_size // 8) * (self.input_size // 8)),
            nn.BatchNorm1d(512 * (self.input_size // 8) * (self.input_size // 8)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 5, 2, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2,2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, self.output_dim, 5, 2, 2, 1),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 512, (self.input_size // 8), (self.input_size // 8))
        
        x = self.deconv(x)

        return x
    
class Discriminator_cat(nn.Module):
    def __init__(self,y_dim,logits):
        super(Discriminator_cat, self).__init__()
        self.feat_net = nn.Sequential(
            nn.Linear(y_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),

        )
        self.logits=logits
        self.out_net= nn.Sequential(
            nn.Linear(1000, 1),
        )
        self.apply(weights_init)

    def forward(self, z):
        feat = self.feat_net(z)
        out=self.out_net(feat)
        if self.logits:
            return [feat,out]
        else:
            return [feat,torch.sigmoid(out)]
    
class Discriminator_gauss(nn.Module):
    def __init__(self,z_dim,logits):
        super(Discriminator_gauss, self).__init__()
        self.feat_net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),

        )
        self.logits=logits
        self.out_net= nn.Sequential(
            nn.Linear(1000, 1),
        )
        self.apply(weights_init)

    def forward(self, z):
        feat = self.feat_net(z)
        out=self.out_net(feat)
        if self.logits:
            return [feat,out]
        else:
            return [feat,torch.sigmoid(out)]

