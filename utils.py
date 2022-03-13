import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torchvision import datasets,transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torchvision,torch

import datasets,losses

def get_categorical(label, y_dim):
    latent_y = torch.Tensor(np.eye(y_dim)[label].astype('float32'))
    p=torch.mean(latent_y,dim=0).cuda()
    return p

#sampling from the categorical distribution based on the known bag-level ratio
def sampler_real_cat(batch_size,y_dim,p=None):
    p=np.array(p.cpu().data.numpy())
    cat = np.random.choice(range(y_dim), size=batch_size, p=p)
    cat = np.eye(y_dim)[cat].astype('float32')
    cat = torch.Tensor(cat).cuda()
    return cat

def gaussian_mixture(batch_size, n_dim, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * np.cos(r) - y * np.sin(r)
        new_y = x * np.sin(r) + y * np.cos(r)
        new_x += shift * np.cos(r)
        new_y += shift * np.sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, n_dim // 2))
    y = np.random.normal(0, y_var, (batch_size, n_dim // 2))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim // 2):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

    return z

def swiss_roll(batch_size, n_dim, n_labels=10, label_indices=None):
    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = np.sqrt(uni) * 3.0
        rad = np.pi * 5.0 * np.sqrt(uni)
        x = r * np.cos(rad)
        y = r * np.sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim // 2):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, n_labels), n_labels)
    return z

def normal(batch_size, z_dim):
    z=np.random.normal(size=(batch_size,z_dim))
    return z
    

def eval_encoder(En,De,dataloader_test,dataset_name,method):
    latent_z,fake_labels=[],[]
    err_epoch_test=0
    entropy_instance_level_epoch_test=[]
    cross_entropy_instance_level_epoch_test=[]
    reconstruct_loss_epoch_test=[]
    epsilon=1e-8
    
    dataloader_train_len,dataloader_test_len=datasets.load_length(dataset_name)
    with torch.no_grad():
        acc_total_test=0
        for i, (imgs, labels) in enumerate(dataloader_test):
            
            imgs=imgs.cuda()
            fake_cat,fake_gauss = En(imgs)
            z=fake_gauss.detach().cpu().data.numpy()
            label_pred=fake_cat.detach().cpu().data.numpy()
            
            cross_entropy_instance_level=0
            for j in range(imgs.shape[0]):
                index=np.where(np.max(label_pred[j])==label_pred[j])[0][0]
                latent_z.append(z[j])
                if labels[j]==index:
                    acc_total_test+=1
                if method=='llp_aae':
                    fake_labels.append(index)
                elif method=='ss_aae':
                    fake_labels.append(labels[j])
                cross_entropy_instance_level+=-np.log(np.clip(label_pred[j][labels[j]], epsilon, 1-epsilon))
            cross_entropy_instance_level_epoch_test.append(cross_entropy_instance_level)
            
            #entropy instance-level on testing data
            ent_test=losses.cross_entropy_loss(fake_cat,fake_cat)
            ent_instance_level=torch.sum(ent_test,dim=1).sum()
            entropy_instance_level_epoch_test.append(ent_instance_level.item())
            #calculate the reconstruction loss on training data
            recon_imgs=De(torch.cat((fake_cat,fake_gauss),dim=1))
            recon_loss_=torch.mean((imgs-recon_imgs)**2)
            reconstruct_loss_epoch_test.append(recon_loss_.item())
            
            #save the imgs and recon_imgs
            save_image(imgs*0.5+0.5,'./generated samples/reconstruct/real_imgs.png',nrow=20)
            save_image(recon_imgs*0.5+0.5,'./generated samples/reconstruct/recon_imgs.png',nrow=20)
                
        err_epoch_test=1-acc_total_test/dataloader_test_len
        latent_z,fake_labels=np.array(latent_z),np.array(fake_labels)
                
        return err_epoch_test,latent_z,fake_labels,\
               entropy_instance_level_epoch_test,\
               cross_entropy_instance_level_epoch_test,reconstruct_loss_epoch_test

def save_latent_variable(dataset_name,method,sample, hy, batch_size, epoch, latent_values, labels,z_dim):
    if (epoch+1)%10==0:
        #c=['b','g','r','c','m','y','k','olive','gray','brown']
        c=['dodgerblue','g','r','c','m','y','k','lime','gray','orange']
        import matplotlib.patches as patches
        rects=[patches.Rectangle((0,0),1,1,facecolor=i) for i in c]
        latent_values=latent_values*5
        plt.figure(figsize=(8, 8))
        for i in set(labels):
            plt.scatter(latent_values[labels == i][:, 0], latent_values[labels == i][:, 1], label=str(i),s=5,marker='o',c=c[i])
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        plt.legend(rects, [str(i) for i in range(10)],loc="upper right")
        plt.savefig('./index/'+ dataset_name +'/'+str(hy) +'_'+str(batch_size)+'_'+str(epoch)+method+sample+'.png')
        plt.close()
    
def plot_loss(algorithm,sample,z_dim, dataset_name, count, batch_size, hy, xtick,y1,y2,y3,y4,y5,y6,y7):
    #save the data of index info
    if algorithm=='llp_aae':
        np.save('./index/'+ dataset_name + '/'+sample+ ' '+str(z_dim)+' entropy_instance_level_epoch_test '+str(hy)+ '_'+str(batch_size)+'.npy',y1)
        np.save('./index/'+ dataset_name + '/'+sample+ ' '+str(z_dim)+' cross_entropy_instance_level_epoch_test '+str(hy)+ '_'+str(batch_size)+'.npy',y2)
        np.save('./index/'+ dataset_name + '/'+sample+ ' '+str(z_dim)+' reconstruct_loss_epoch_test '+str(hy)+ '_'+str(batch_size)+'.npy',y3)
        np.save('./index/'+ dataset_name+ '/'+sample+ ' '+str(z_dim) +' err_epoch_test '+str(hy)+ '_'+str(batch_size)+'.npy',y4)
    
        #plt.clf()
        plt.figure(figsize=(5,8))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=0.45)
    
        plt.subplot(511)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title(dataset_name +' '+sample+''+str(z_dim)+' bagsize=' + str(batch_size)+' hy='+ str(hy) +' entropy instance-level test',fontsize=10)
        plt.plot(xtick[:count],y1[:count],color='red')
        
        plt.subplot(512)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title(' cross entropy instance-level test(green)',fontsize=10)
        plt.plot(xtick[:count],y2[:count],color='green')
    
        plt.subplot(513)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title('reconstruct_loss test ',fontsize=10)
        plt.plot(xtick[:count],y3[:count],color='red')
    
        plt.subplot(514)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title('error rate on test',fontsize=10)
        plt.yticks(np.arange(0,1.1,0.1))
        plt.ylim([0,1])
        plt.plot(xtick[:count],y4[:count],color='red')
        
        plt.savefig('./images/'+ dataset_name + '/' +algorithm + ' '+sample+ ' '+str(z_dim)+' loss bagsize=%d hy=%.2f.png'%(batch_size, hy ))
        plt.close()
        
    elif algorithm=='llp_gan':
        np.save('./index/'+ dataset_name + '/'+sample+ ' '+str(z_dim)+' entropy_instance_level_epoch_test '+str(hy)+ '_'+str(batch_size)+'gan.npy',y1)
        np.save('./index/'+ dataset_name + '/'+sample+ ' '+str(z_dim)+' cross_entropy_instance_level_epoch_test '+str(hy)+ '_'+str(batch_size)+'gan.npy',y2)
        np.save('./index/'+ dataset_name + '/'+sample+ ' '+str(z_dim)+' reconstruct_loss_epoch_test '+str(hy)+ '_'+str(batch_size)+'gan.npy',y3)
        np.save('./index/'+ dataset_name+ '/'+sample+ ' '+str(z_dim) +' err_epoch_test '+str(hy)+ '_'+str(batch_size)+'gan.npy',y4)
    
        #plt.clf()
        plt.figure(figsize=(5,8))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=0.45)
    
        plt.subplot(511)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title(dataset_name +' '+sample+''+str(z_dim)+' bagsize=' + str(batch_size)+' hy='+ str(hy) +' entropy instance-level test',fontsize=10)
        plt.plot(xtick[:count],y1[:count],color='red')
        
        plt.subplot(512)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title(' cross entropy instance-level test(green)',fontsize=10)
        plt.plot(xtick[:count],y2[:count],color='green')
    
        plt.subplot(513)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title('reconstruct_loss test ',fontsize=10)
        plt.plot(xtick[:count],y3[:count],color='red')
    
        plt.subplot(514)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title('error rate on test',fontsize=10)
        plt.yticks(np.arange(0,1.1,0.1))
        plt.ylim([0,1])
        plt.plot(xtick[:count],y4[:count],color='red')
        
        plt.savefig('./images/'+ dataset_name + '/' +algorithm + ' '+sample+ ' '+str(z_dim)+' loss bagsize=%d hy=%.2f gan.png'%(batch_size, hy ))
        plt.close()
    
    elif algorithm=='dllp':
        np.save('./index/'+ dataset_name +'/entropy_instance_level_epoch_test '+str(hy)+ '_'+str(batch_size)+algorithm,y2)
        np.save('./index/'+ dataset_name +'/cross_entropy_instance_level_epoch_test '+str(hy)+ '_'+str(batch_size)+algorithm,y3)
        np.save('./index/'+ dataset_name +'/err_epoch_test '+str(hy)+ '_'+str(batch_size)+algorithm,y5)
        
        plt.figure(figsize=(5,8))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=0.45)

        plt.subplot(511)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title(' entropy instance-level test',fontsize=10)
        plt.plot(xtick[:count],y1[:count],color='red')
        
        plt.subplot(512)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title(' cross entropy instance-level test(green)',fontsize=10)
        plt.plot(xtick[:count],y2[:count],color='green')
        
        plt.subplot(513)
        ax = plt.gca() 
        ax.tick_params(labelright = True)
        ax.grid()
        plt.title('error rate on test',fontsize=10)
        plt.yticks(np.arange(0,1.1,0.1))
        plt.ylim([0,1])
        plt.plot(xtick[:count],y4[:count],color='red')

    
        plt.savefig('./images/'+ dataset_name + '/' +algorithm +'loss bagsize=%d hy=%.2f.png'%(batch_size, hy ))
        plt.close()
        
        
#def plot_ent():