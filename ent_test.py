import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torchvision import datasets,transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam

import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torchvision,torch
import torch.autograd as autograd

#import libraries self-constructed
import datasets,losses,utils
from networks import Encoder_grey,Decoder_grey,Encoder_RGB,Decoder_RGB,Discriminator_cat,Discriminator_gauss

def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate
    if epoch >= 100:
        lr = learning_rate * (0.5 ** (epoch // 100))  # i.e. 240,320
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def zero_grad_all(optimizer_En,optimizer_De,optimizer_D_cat,optimizer_D_gauss):
    optimizer_En.zero_grad()
    optimizer_De.zero_grad()
    optimizer_D_cat.zero_grad()
    optimizer_D_gauss.zero_grad()

def train_start(args):
    dataset_name=args.dataset_name
    batch_size=args.batch_size
    y_dim=args.y_dim
    z_dim=args.z_dim
    lr=args.lr
    n_epochs=args.n_epochs
    img_size=args.img_size
    channels=args.channels
    hy=args.hyperparameter
    beta1=args.beta1
    beta2=args.beta2
    betas=(beta1,beta2)
    method=args.method
    data_augmentation=args.data_augmentation
    eps=args.eps
    sample=args.sample
    ent=args.ent
    
    #configure the environment and gpu
    #cuda=False
    cuda = True if torch.cuda.is_available() else False
    device=torch.device('cuda' if cuda else 'cpu')
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    #configure data
    dataloader_train,dataloader_test=datasets.load_data_from(method, data_augmentation, dataset_name, batch_size)
    dataloader_train_len,dataloader_test_len=datasets.load_length(dataset_name)
    bags=dataloader_train_len//batch_size#the number of total bags of equal size on the training data

    # Initialize generator and discriminator
    if dataset_name in ['mnist','fashion_mnist','kmnist']:
        
        En=Encoder_grey(y_dim,z_dim,method).cuda()
        De=Decoder_grey(y_dim,z_dim,method).cuda()
    elif dataset_name in ['cifar10','svhn']:
        En=Encoder_RGB(y_dim,z_dim,method).cuda()
        De=Decoder_RGB(y_dim,z_dim,method).cuda()
    D_cat=Discriminator_cat(y_dim,logits=True).cuda()
    
    path='./state_dict/ent/'+ dataset_name + '/encoder '+str(hy)+' '+str(batch_size)+str(ent)+'.pth'
    En.load_state_dict(torch.load(path))
    
    D_gauss=Discriminator_gauss(z_dim,logits=True).cuda()
    
    #initialize networks
    #path='./state_dict/'+ dataset_name + '/encoder '+str(100)+' '+str(64)+'.pth'
    #En.load_state_dict(torch.load(path))

    
    xtick,y1,y2,y3,y4,y5,y6,y7=[],[],[],[],[],[],[],[]
    
    for epoch in range(1):
        
        ent_list=[]
        cnt=0
        
        for i, (imgs, labels) in enumerate(dataloader_train):
            # Configure input
            real_imgs = imgs.cuda() 
            for x in real_imgs:
                #fake label's entropy
                fake_cat_, _=En(real_imgs)
                ent_tmp=losses.cross_entropy_loss(fake_cat_,fake_cat_)
                ent_list.append(torch.sum(ent_tmp,dim=1).mean().item())
        #plot the self entropy of training data
        plt.figure(figsize=(8, 8))
        #plt.scatter(np.arange(cnt),ent_list,s=5,marker='o')
        plt.xlim(np.min(ent_list),np.max(ent_list))
        #plt.xlim(2.2985,np.max(ent_list))
        plt.hist(ent_list,bins=100,edgecolor='k')
        mean=round(np.mean(ent_list),4);std=round(np.std(ent_list),4);median=round(np.median(ent_list),4)
        ax=plt.gca()
        plt.text(1,1,va='top',ha='right',s='mean='+str(mean)+'\nstd='+str(std)+'\nmedian='+str(median),transform=ax.transAxes,fontsize=20)
        plt.xticks(size=18)
        plt.yticks(size=18)
        
        #plt.ylim(0, 40)
        plt.savefig('./images/'+ dataset_name +'/'+str(hy) +'_'+str(batch_size)+'_'+str(epoch)+method+sample+str(ent)+'train.png')
        plt.close()
