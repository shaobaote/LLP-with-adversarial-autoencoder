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

#import libraries self-constructed
import networks,datasets,losses,utils,train_ss_aae,train_aae,train_dllp,SL



def get_args():
    dataset_name='kmnist'
    batch_size=32
    method='llp_aae'
    sample='normal'
    data_augmentation=False
    hy=300
    
    if method=='dllp':data_augmentation=True; hy=1
    y_dim=10 if dataset_name not in ['cifar100'] else 100 #number of classes for the data
    z_dim= 2
    img_size=32 if dataset_name in ['svhn','cifar10'] else 28  #length of size for the image
    channels=3 if dataset_name in ['svhn','cifar10'] else 1
    
    parser = argparse.ArgumentParser("Learning from Label Proportions with Adversarial Autoencoder")

    # basic arguments
    parser.add_argument("--dataset_name", type=str,default=dataset_name)
    parser.add_argument("--sample", type=str,default=sample)
    parser.add_argument("--batch_size", default=batch_size, type=int)
    parser.add_argument("--method", type=str,default=method)
    parser.add_argument("--data_augmentation", type=str,default=data_augmentation)
    #parser.add_argument("--model_name", type=str, default="wrn28-2")
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--metric", type=str, default="ce")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--y_dim", default=y_dim, type=int)
    parser.add_argument("--z_dim", default=z_dim, type=int)
    parser.add_argument("--beta1", default=0.5, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--img_size", default=img_size, type=int)
    parser.add_argument("--channels", default=channels, type=int)
    
    # coefficient for balance fm-loss and prop-loss, hy is for fm-loss
    parser.add_argument("--hyperparameter", default=hy, type=int)

    return parser.parse_args()

def main(args):
    if args.method=='dllp':
        train_dllp.train_start(args)
    elif args.method=='llp_aae':
        SL.train_start(args)
    elif args.method=='ss_aae':
        train_ss_aae.train_start(args)
    elif args.method=='aae':
        train_aae.train_start(args)
    else:
        print('method isn\'t supported')

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed(args.seed)
    main(args)

