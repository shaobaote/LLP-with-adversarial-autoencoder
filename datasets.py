import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torchvision import datasets,transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, BatchSampler, RandomSampler
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torchvision,torch

# Configure data floder
os.makedirs("../dataset/mnist", exist_ok=True)
os.makedirs("../dataset/cifar10", exist_ok=True)
os.makedirs("../dataset/svhn", exist_ok=True)
os.makedirs("../dataset/fashion_mnist", exist_ok=True)
os.makedirs("../dataset/kmnist", exist_ok=True)
os.makedirs("../dataset/cifar100", exist_ok=True)
os.makedirs("./images/cifar10", exist_ok=True)
os.makedirs("./images/mnist", exist_ok=True)
os.makedirs("./images/svhn", exist_ok=True)
os.makedirs("./images/cifar100", exist_ok=True)
os.makedirs("./images/fashion_mnist", exist_ok=True)
os.makedirs("./images/kmnist", exist_ok=True)
os.makedirs("./index/mnist", exist_ok=True)
os.makedirs("./index/fashion_mnist", exist_ok=True)
os.makedirs("./index/cifar10", exist_ok=True)
os.makedirs("./index/cifar100", exist_ok=True)
os.makedirs("./index/svhn", exist_ok=True)
os.makedirs("./index/kmnist", exist_ok=True)
os.makedirs("./state_dict/mnist", exist_ok=True)
os.makedirs("./state_dict/fashion_mnist", exist_ok=True)
os.makedirs("./state_dict/cifar10", exist_ok=True)
os.makedirs("./state_dict/cifar100", exist_ok=True)
os.makedirs("./state_dict/svhn", exist_ok=True)
os.makedirs("./state_dict/kmnist", exist_ok=True)


datasets_name_list=['mnist','fashion_mnist','svhn','cifar10','cifar100']
img_size_RGB=32
img_size_grey=28

transform_RGB_train_augmentation=transforms.Compose(
    [transforms.Resize(img_size_RGB),
     transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
transform_RGB_train=transforms.Compose(
    [transforms.Resize(img_size_RGB),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
transform_RGB_test=transforms.Compose(
    [transforms.Resize(img_size_RGB),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

transform_grey_train_augmentation=transforms.Compose(
    [transforms.Resize(img_size_grey),
     transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(28, padding=4),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5],std=[0.5])])
transform_grey_train=transforms.Compose(
    [transforms.Resize(img_size_grey),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5],std=[0.5])])
transform_grey_test=transforms.Compose(
    [transforms.Resize(img_size_grey),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5],std=[0.5])])

class BagSampler(Sampler):
    def __init__(self, bags, num_bags=-1):
        """
        params:
            bags: shape (num_bags, num_instances), the element of a bag
                  is the instance index of the dataset
            num_bags: int, -1 stands for using all bags
        """
        self.bags = bags
        if num_bags == -1:
            self.num_bags = len(bags)
        else:
            self.num_bags = num_bags
        assert 0 < self.num_bags <= len(bags)

    def __iter__(self):
        indices = torch.randperm(self.num_bags)
        for index in indices:
            yield self.bags[index]

    def __len__(self):
        return len(self.bags)

#return training and testing set in minibatch
def load_mnist(data_augmentation, batch_size):
    if data_augmentation==True:
        train_set=datasets.MNIST(root='../dataset/mnist',train=True,download=True,transform=transform_grey_train_augmentation)
    else:
        train_set=datasets.MNIST(root='../dataset/mnist',train=True,download=True,transform=transform_grey_train)
    test_set=datasets.MNIST(root='../dataset/mnist',train=False,download=True,transform=transform_grey_test)
    return train_set,test_set

def load_kmnist(data_augmentation, batch_size):
    if data_augmentation==True:
        train_set=datasets.KMNIST(root='../dataset/kmnist',train=True,download=True,transform=transform_grey_train_augmentation)
    else:
        train_set=datasets.KMNIST(root='../dataset/kmnist',train=True,download=True,transform=transform_grey_train)
    test_set=datasets.KMNIST(root='../dataset/kmnist',train=False,download=True,transform=transform_grey_test)
    return train_set,test_set

def load_fashion_mnist(data_augmentation, batch_size):
    if data_augmentation==True:
        train_set=datasets.FashionMNIST(root='../dataset/fashion_mnist',train=True,download=True,transform=transform_grey_train_augmentation)
    else:
        train_set=datasets.FashionMNIST(root='../dataset/fashion_mnist',train=True,download=True,transform=transform_grey_train)
    test_set=datasets.FashionMNIST(root='../dataset/fashion_mnist',train=False,download=True,transform=transform_grey_test)
    return train_set,test_set

def load_cifar10(data_augmentation, batch_size):
    if data_augmentation==True:
        train_set=datasets.CIFAR10(root='../dataset/cifar10',train=True,download=True,transform=transform_RGB_train_augmentation)
    else:
        train_set=datasets.CIFAR10(root='../dataset/cifar10',train=True,download=True,transform=transform_RGB_train)
    test_set=datasets.CIFAR10(root='../dataset/cifar10',train=False,download=True,transform=transform_RGB_test)
    return train_set,test_set

def load_svhn(data_augmentation, batch_size):
    if data_augmentation==True:
        train_set=datasets.SVHN(root='../dataset/svhn',split='train',download=True,transform=transform_RGB_train_augmentation)
    else:
        train_set=datasets.SVHN(root='../dataset/svhn',split='train',download=True,transform=transform_RGB_train)
    test_set=datasets.SVHN(root='../dataset/svhn',split='test',download=True,transform=transform_RGB_test)
    return train_set,test_set

def load_cifar100(data_augmentation, batch_size):
    if data_augmentation==True:
        train_set=datasets.CIFAR100(root='../dataset/cifar100',train=True,download=True,transform=transform_RGB_train_augmentation)
    else:
        train_set=datasets.CIFAR100(root='../dataset/cifar100',train=True,download=True,transform=transform_RGB_train)
    test_set=datasets.CIFAR100(root='../dataset/cifar100',train=False,download=True,transform=transform_RGB_test)
    return train_set,test_set

def load_length(dataset_name):
    if dataset_name in ['mnist','fashion_mnist','kmnist']:
        return 60000,10000
    if dataset_name in ['cifar10','cifar100']:
        return 50000,10000
    elif dataset_name=='svhn':
        dataset_train=datasets.SVHN("../dataset/svhn",split='train',download=True,transform=transform_RGB_train)
        dataset_test=datasets.SVHN("../dataset/svhn",split='test',download=True,transform=transform_RGB_test)
    return len(dataset_train),len(dataset_test)

def load_data_from(method, data_augmentation, dataset_name,batch_size):
    if dataset_name=='mnist':
        train_set,test_set =load_mnist(data_augmentation, batch_size)
    elif dataset_name=='fashion_mnist':
        train_set,test_set =load_fashion_mnist(data_augmentation, batch_size)
    elif dataset_name=='kmnist':
        train_set,test_set =load_kmnist(data_augmentation, batch_size)
    elif dataset_name=='svhn':
        train_set,test_set =load_svhn(data_augmentation, batch_size)
    elif dataset_name=='cifar10':
        train_set,test_set =load_cifar10(data_augmentation,batch_size)
    elif dataset_name=='cifar100':
        train_set,test_set =load_cifar100(data_augmentation, batch_size)
        
    len_train, _=load_length(dataset_name)
    indices = RandomSampler(range(len_train), replacement=False)
    bags = list(BatchSampler(indices, batch_size=batch_size,drop_last=True))
    train_bag_sampler = BagSampler(bags)
    train_dataloader = DataLoader(train_set, batch_sampler=train_bag_sampler, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=200, shuffle=False, pin_memory=True, num_workers=4,drop_last=False)
    
    return train_dataloader, test_dataloader
