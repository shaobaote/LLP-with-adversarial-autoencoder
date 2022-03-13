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
        adjust_learning_rate(optimizer_En, epoch, lr)
        adjust_learning_rate(optimizer_De, epoch, lr)
        adjust_learning_rate(optimizer_D_cat, epoch, lr)
        adjust_learning_rate(optimizer_D_gauss, epoch, lr)
        
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
            if sample=='swiss_roll':
                real_gauss=Tensor(utils.swiss_roll(batch_size, z_dim)).cuda()
            elif sample=='gaussian_mixture':
                real_gauss=Tensor(utils.gaussian_mixture(batch_size, z_dim)).cuda()
            elif sample=='normal':
                real_gauss=Tensor(utils.normal(batch_size, z_dim)).cuda()
            else:
                print('error')
            real_prop=utils.get_categorical(labels,y_dim)
            real_cat=utils.sampler_real_cat(batch_size,y_dim,real_prop)
            
            #######reconstruct phase#######
            zero_grad_all(optimizer_En,optimizer_De,optimizer_D_cat,optimizer_D_gauss)

            recon_imgs=De(torch.cat(En(real_imgs),dim=1))
            recon_loss=torch.mean((real_imgs-recon_imgs)**2)

            if epoch>0:
                recon_loss.backward()
                optimizer_En.step()
                optimizer_De.step()
        
            #######train discriminator#######
            zero_grad_all(optimizer_En,optimizer_De,optimizer_D_cat,optimizer_D_gauss)
            fake_cat,fake_gauss = En(real_imgs)      
                
            #WGAN-GP
            gradient_penalty_cat = losses.cal_gradient_penalty(D_cat, 'cuda', real_cat, fake_cat)
            d_loss_cat=1*(-D_cat(real_cat)[1].mean() + D_cat(fake_cat)[1].mean()) + 1* gradient_penalty_cat
            gradient_penalty_gauss = losses.cal_gradient_penalty(D_gauss, 'cuda', real_gauss, fake_gauss)
            d_loss_gauss=1*(-D_gauss(real_gauss)[1].mean() + D_gauss(fake_gauss)[1].mean()) + 1* gradient_penalty_gauss
            #0.2,0.0001
            #GAN
            #d_loss_gauss=-0.2*torch.mean(torch.log(D_gauss(real_gauss)[1] + eps)+torch.log(1-D_gauss(fake_gauss)[1] + eps))
            #d_loss_cat=-0.2*torch.mean(torch.log(D_cat(real_cat)[1] + eps)+torch.log(1-D_cat(fake_cat)[1] + eps))
           
            d_loss=d_loss_cat+d_loss_gauss
            #d_loss=d_loss_gauss
            #d_loss=d_loss_cat
            
            D_gauss_loss.append(d_loss_gauss.item())
            D_cat_loss.append(d_loss_cat.item())
            
            if epoch>0:
                d_loss.backward()
                optimizer_D_gauss.step()
                optimizer_D_cat.step()
        
            #######train generator/encoder#######  
            zero_grad_all(optimizer_En,optimizer_De,optimizer_D_cat,optimizer_D_gauss)
            
            fake_cat,fake_gauss = En(real_imgs)
            
            #fake label's entropy
            ent_tmp=losses.cross_entropy_loss(fake_cat,fake_cat)
            ent_list.append(torch.sum(ent_tmp,dim=1).mean().item())
            
            cnt+=1
            
            fake_prop=torch.mean(fake_cat,dim=0)
            prop_loss=torch.sum(losses.cross_entropy_loss(fake_prop,real_prop), dim=-1).mean()
            #WGAN-GP
            g_loss_adv=-0.1*(D_gauss(fake_gauss)[1].mean()+D_cat(fake_cat)[1].mean())
            #0.02
            #g_loss_adv=-0.02*(D_gauss(fake_gauss)[1].mean())
            #g_loss_adv=-0.02*(D_cat(fake_cat)[1].mean())
            
            #GAN
            #g_loss_adv = - 0.02*torch.mean(torch.log(D_gauss(fake_gauss)[1] + eps))      
            G_loss.append(g_loss_adv.item())
            Prop_loss.append(prop_loss.item())
            
            if epoch>0:
                g_loss=g_loss_adv + hy*prop_loss
                #g_loss= hy*prop_loss
                g_loss.backward()
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
        y3.append(np.sum(reconstruct_loss_epoch_test)*batch_size/dataloader_test_len)
        y4.append(err_epoch_test)
        y5.append(np.mean(G_loss))
        y6.append(np.mean(D_cat_loss))
        y7.append(np.mean(D_gauss_loss))
                
        
        
        #save the figures
        utils.save_latent_variable(dataset_name, method,sample, hy, batch_size, epoch, latent_z, fake_labels,z_dim)
        utils.plot_loss('llp_aae',sample,z_dim,dataset_name, epoch+1, batch_size, hy, xtick, y1,y2,y3,y4,y5,y6,y7)
    
        print('epoch={} error_test={}'.format(epoch,err_epoch_test))
        print('Prop_loss={} G_loss={} D_loss_cat={} D_loss_gauss={}'.format(Prop_loss[-1],G_loss[-1],D_cat_loss[-1],D_gauss_loss[-1]))
    
        #save the model per epoch
        torch.save(En.state_dict(),'./state_dict/'+ dataset_name + '/encoder '+str(hy)+' '+str(batch_size)+'.pth')
        torch.save(De.state_dict(),'./state_dict/'+ dataset_name + '/decoder '+str(hy)+' '+str(batch_size)+'.pth')