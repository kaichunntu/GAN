#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import numpy as np

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as img

if not( os.path.exists("./samples") ):
    os.mkdir("./samples")

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.dense = nn.Sequential( 
            nn.Linear( 100, 8 * 8 * 384 ),
            nn.SELU(),)
        self.dense_bn = nn.BatchNorm1d( 8 * 8 * 384 )
        self.upsampling1 = nn.Sequential(
            nn.ConvTranspose2d( in_channels=384, out_channels=384, kernel_size=8, stride=2, padding = 1 ),
            nn.SELU(),)
        self.conv1 = nn.Sequential(
            nn.Conv2d(384, 256, 5, 1 ), 
            nn.SELU(),)
        self.upsampling2 = nn.Sequential( 
            nn.ConvTranspose2d( 256, 192, 5, 2, ), 
            nn.SELU(),
            nn.BatchNorm1d( 192 ) )
        self.conv2 = nn.Sequential(
            nn.Conv2d( 192, 128, 4, 1 ), 
            nn.SELU(),)
        self.upsampling3 = nn.Sequential( 
            nn.ConvTranspose2d( 128, 64, 5, 2 ), 
            nn.SELU(),)
        self.Image = nn.Sequential(
            nn.Conv2d(64, 3, 4, 1, ), 
            nn.Tanh(),)

            
    def forward(self, z):
        #print( "Generator :" )
        layer = self.dense( z )
        layer = self.dense_bn( layer )
        layer  = layer.view( -1, 384, 8, 8 )
        #print( layer.shape )
        layer  = self.upsampling1( layer )
        #print( layer.shape )
        layer  = self.conv1( layer )
        #print( layer.shape )
        layer  = self.upsampling2( layer )
        #print( layer.shape )
        layer  = self.conv2( layer )
        #print( layer.shape )
        layer  = self.upsampling3( layer )
        #print( layer.shape )
        f_img  = self.Image( layer )
        #print( f_img.shape )
        return f_img
    
    
z_dim = 100
b_size = 100
n = 1


G = generator().cuda()

G.load_state_dict(torch.load('Gen_norm.pkl'))

np.random.seed( 17 )
z = Variable( torch.Tensor( np.random.normal( 0, 1, size = [ b_size , z_dim ]) ) ).cuda()
fake_img = G( z )
fake_img = np.transpose( ((fake_img.cpu().data.numpy()+1)*128).astype(int), [0,2,3,1])   
fig, axs = plt.subplots(5, 5)
cnt = 0
for i in range(5):
    for j in range(5):
        axs[i,j].imshow(fake_img[cnt, :,:,:])
        axs[i,j].axis('off')
        cnt += 1
        fig.savefig( "samples/gan.png" )
    plt.close()

