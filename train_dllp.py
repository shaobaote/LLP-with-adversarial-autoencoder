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
from networks import Encoder_grey,Encoder_RGB

def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate
    if epoch >= 100:
        lr = learning_rate * (0.5 ** (epoch // 100))  # i.e. 240,320
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
    sample=''
    
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
        En=Encoder_grey(y_dim,z_dim,method='dllp').cuda()
    elif dataset_name in ['cifar10','svhn']:
        En=Encoder_RGB(y_dim,z_dim,method='dllp').cuda()
    
    #configure optimizer
    optimizer_En = Adam(En.parameters(), lr=0.0003,betas=betas)
    
    xtick,y1,y2,y3,y4,y5,y6,y7=[],[],[],[],[],[],[],[]
    
    for epoch in range(n_epochs):
        adjust_learning_rate(optimizer_En, epoch, lr)
        
        err_epoch_test=0
        entropy_instance_level_epoch_test=[]
        cross_entropy_instance_level_epoch_test=[]
        Prop_loss=[]
        
        for i, (imgs, labels) in enumerate(dataloader_train):
            # Configure input
            real_imgs = imgs.cuda()  
            real_prop=utils.get_categorical(labels,y_dim)

            optimizer_En.zero_grad()
            
            fake_cat = En(real_imgs)
            fake_prop=torch.mean(fake_cat,dim=0)
            prop_loss=torch.sum(losses.cross_entropy_loss(fake_prop,real_prop), dim=-1).mean()
            
            Prop_loss.append(prop_loss.item())
            
            if epoch>0:
                prop_loss.backward()
                optimizer_En.step()  
        
        #evaluate on test
        with torch.no_grad():
            acc_total_test=0
            for i, (imgs, labels) in enumerate(dataloader_test):
                imgs=imgs.cuda()
                fake_cat = En(imgs)
                label_pred=fake_cat.detach().cpu().data.numpy()
            
                cross_entropy_instance_level=0
                for j in range(imgs.shape[0]):
                    index=np.where(np.max(label_pred[j])==label_pred[j])[0][0]
                    if labels[j]==index:
                        acc_total_test+=1
                    cross_entropy_instance_level+=-np.log(np.clip(label_pred[j][labels[j]], eps, 1-eps))
                cross_entropy_instance_level_epoch_test.append(cross_entropy_instance_level)
            err_epoch_test=1-acc_total_test/dataloader_test_len
            
            ent_test=losses.cross_entropy_loss(fake_cat,fake_cat)
            ent_instance_level=torch.sum(ent_test,dim=1).sum()
            entropy_instance_level_epoch_test.append(ent_instance_level.item())
           
        xtick.append(epoch)
        y1.append(np.sum(entropy_instance_level_epoch_test)/dataloader_test_len)
        y2.append(np.sum(cross_entropy_instance_level_epoch_test)/dataloader_test_len)
        y4.append(err_epoch_test)
        
        #save the figures
        utils.plot_loss('dllp',sample,z_dim,dataset_name, epoch+1, batch_size, hy, xtick, y1,y2,y3,y4,y5,y6,y7)
    
        print('epoch={} error_test={}'.format(epoch,err_epoch_test))
        print('Prop_loss={}'.format(Prop_loss[-1]))
