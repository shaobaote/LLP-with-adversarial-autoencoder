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

def adjust_learning_rate(optimizer, epoch, learning_rate,batch_size):
    lr=learning_rate
    if epoch>=100:
        lr = learning_rate * (0.5 ** (epoch // 100))
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
    elif dataset_name in ['cifar10','svhn','cifar100']:
        En=Encoder_RGB(y_dim,z_dim,method).cuda()
        De=Decoder_RGB(y_dim,z_dim,method).cuda()
    D_cat=Discriminator_cat(y_dim,logits=True).cuda()
    D_gauss=Discriminator_gauss(z_dim,logits=True).cuda()
    
    #initialize networks
    #path='./state_dict/'+ dataset_name + '/encoder '+str(100)+' '+str(64)+'.pth'
    #En.load_state_dict(torch.load(path))
    
    #configure optimizer
    optimizer_En = Adam(En.parameters(), lr=3e-4,betas=betas)
    optimizer_De = Adam(De.parameters(), lr=3e-4,betas=betas)
    optimizer_D_cat = Adam(D_cat.parameters(),lr=3e-4,betas=betas)
    optimizer_D_gauss = Adam(D_gauss.parameters(),lr=3e-4,betas=betas)
    
    xtick,y1,y2,y3,y4,y5,y6,y7=[],[],[],[],[],[],[],[]
    
    for epoch in range(n_epochs):
        adjust_learning_rate(optimizer_En, epoch, lr,batch_size)
        adjust_learning_rate(optimizer_De, epoch, lr,batch_size)
        adjust_learning_rate(optimizer_D_cat, epoch, lr,batch_size)
        adjust_learning_rate(optimizer_D_gauss, epoch, lr,batch_size)
        
        err_epoch_test=0
        entropy_instance_level_epoch_test=[]
        cross_entropy_instance_level_epoch_test=[]
        reconstruct_loss_epoch_test=[]
        Prop_loss=[]
        G_loss=[]
        D_cat_loss=[]
        D_gauss_loss=[]
        
        ent_list=[]
        cnt=0
        
        for i, (imgs, labels) in enumerate(dataloader_train):
            # Configure input
            real_imgs = imgs.cuda()
            labels=labels.cuda()
            if sample=='swiss_roll':
                real_gauss=Tensor(utils.swiss_roll(batch_size, z_dim)).cuda()
            elif sample=='gaussian_mixture':
                real_gauss=Tensor(utils.gaussian_mixture(batch_size, z_dim)).cuda()
            elif sample=='normal':
                real_gauss=Tensor(utils.normal(batch_size, z_dim)).cuda()
            else:
                print('error')    
        
            #######train generator/encoder#######  
            zero_grad_all(optimizer_En,optimizer_De,optimizer_D_cat,optimizer_D_gauss)
            
            fake_cat,fake_gauss = En(real_imgs)
            
            #fake label's entropy
            
            cnt+=1
            
            loss=F.nll_loss(fake_cat,labels)
            #WGAN-GP
            #0.02
            #g_loss_adv=-0.02*(D_gauss(fake_gauss)[1].mean())
            #g_loss_adv=-0.02*(D_cat(fake_cat)[1].mean())
            
            #GAN
            #g_loss_adv = - 0.02*torch.mean(torch.log(D_gauss(fake_gauss)[1] + eps))      
            
            if epoch>0:
                #g_loss= hy*prop_loss
                loss.backward()
                optimizer_En.step()  
        
        #plot the self entropy of training data
        '''plt.figure(figsize=(8, 8))
        #plt.scatter(np.arange(cnt),ent_list,s=5,marker='o')
        plt.hist(ent_list,bins=20)
        #plt.xlim(0, 40)
        #plt.ylim(0, 40)
        plt.savefig('./images/'+ dataset_name +'/'+str(hy) +'_'+str(batch_size)+'_'+str(epoch)+method+sample+'.png')
        plt.close()'''
        
        #evaluate on test
        err_epoch_test,latent_z,fake_labels,\
        entropy_instance_level_epoch_test,\
        cross_entropy_instance_level_epoch_test,reconstruct_loss_epoch_test\
        =utils.eval_encoder(En,De,dataloader_test,dataset_name, method)
           
        xtick.append(epoch)
        y1.append(np.sum(entropy_instance_level_epoch_test)/dataloader_test_len)
        y2.append(np.sum(cross_entropy_instance_level_epoch_test)/dataloader_test_len)
        y4.append(err_epoch_test)

                
        
        
        #save the figures
        #if dataset_name!='cifar100':
         #   utils.save_latent_variable(dataset_name, method,sample, hy, batch_size, epoch, latent_z, fake_labels,z_dim)
        #utils.plot_loss('llp_aae',sample,z_dim,dataset_name, epoch+1, batch_size, hy, xtick, y1,y2,y3,y4,y5,y6,y7)
    
        print('epoch={} error_test={}'.format(epoch,err_epoch_test))
    
        #save the model per epoch
        #torch.save(En.state_dict(),'./state_dict/'+ dataset_name + '/encoder '+str(hy)+' '+str(batch_size)+'.pth')
        #torch.save(De.state_dict(),'./state_dict/'+ dataset_name + '/decoder '+str(hy)+' '+str(batch_size)+'.pth')