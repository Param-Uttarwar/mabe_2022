import os
import sys
sys.path.append(os.getcwd())
from os.path import join
import torch
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from helper.pointnet.data_prep import mousePCA
import itertools
import pickle
import random

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

class Mousedataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,device):
        self.dataset = np.load(os.path.join(os.getcwd(),'data','pointnet','dataset.npy'))
        self.scaler = pickle.load(open(join(os.getcwd(),'data','pointnet','scaler.pkl'),'rb'))
    def __len__(self):
        return self.dataset.shape[0]*self.dataset.shape[1]
    def __getitem__(self, idx):
        #Get data
        seq,frame = idx//self.dataset.shape[1], idx%self.dataset.shape[1]
        data = self.dataset[seq,frame]
        datashape = data.shape
        #anchor point
        data = self.scaler.transform(data.reshape((-1,60)))
        data = data.reshape(datashape)
        #pos data
        rand_idx = np.random.randint(5,10)*random.choice([-1,1])
        rand_idx = np.clip(rand_idx,0,datashape[1]-1)
        pos_data = self.dataset[seq, rand_idx]
        pos_data = self.scaler.transform(pos_data.reshape((-1,60)))
        pos_data = pos_data.reshape(datashape)
        #neg data
        rand_idx = np.random.randint(0,self.dataset.shape[1]*self.dataset.shape[0]-1)
        neg_data = self.dataset[rand_idx//self.dataset.shape[1],rand_idx%self.dataset.shape[1]]
        neg_data = self.scaler.transform(neg_data.reshape((-1,60)))
        neg_data = neg_data.reshape(datashape)

        #Transform
        data = torch.from_numpy(data.transpose().astype(np.float32)).to(device)
        pos_data = torch.from_numpy(pos_data.transpose().astype(np.float32)).to(device)
        neg_data = torch.from_numpy(neg_data.transpose().astype(np.float32)).to(device)
    
        return [data,pos_data,neg_data]

if __name__ == '__main__':
    dataset = Mousedataset(device)
    item = dataset[np.random.randint(len(dataset))]
    import pdb;pdb.set_trace()