import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
sys.path.append(os.getcwd())
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle

# sys.path.append('/data/param/AIcrowd/mouse')

from helper.pointnet.data_prep import extract_features_from_frames,mousePCA
from helper.pointnet.dataset import Mousedataset
from helper.pointnet.network import PointNetCls


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

class model(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        self.device = self.config['device']
        #Network
        self.net = PointNetCls(k=self.config['out_features'],
                            in_features = self.config['in_channels'],
                         feature_transform=self.config['feature_transform'],
                         global_feat=self.config['global_feat']).to(self.device)
        #Dataset
        self.mousePCA = mousePCA()
        self.dataset = Mousedataset(self.device)
        self.dataloader = DataLoader(self.dataset, batch_size=512, shuffle=False)
        #Loss funtion
        self.loss_fn = nn.CosineEmbeddingLoss().to(device)
        # self.loss_fn = nn.TripletMarginLoss().to(device)
        self.scaler = pickle.load(open(os.path.join(os.getcwd(),'data','pointnet','scaler.pkl'),'rb'))
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.config['learning_rate'])
        self.writer = SummaryWriter(os.path.join(os.getcwd(),'helper/pointnet/checkpoints',self.config['exp_name']))
   
        if self.config['load_path'] is not None:
            self.net.load_state_dict(torch.load(self.config['load_path'] ))
            print('Weights loaded from %s successfully'%self.config['load_path'])

    def calc_loss(self):
        """ Calculate the loss given the predicted and target points 

        Returns
        -------
        self.loss : torch.Tensor
            _description_
        """
        self.pos_loss = self.loss_fn(self.pred,self.pos_pred,torch.ones(self.pred.shape[0]).to(self.device))
        self.neg_loss = self.loss_fn(self.pred,self.neg_pred,(torch.ones(self.pred.shape[0])*-1).to(self.device))
        self.loss = self.pos_loss + self.neg_loss
        # self.loss = self.loss_fn(self.pred,self.pos_pred,self.neg_pred)
        # import pdb;pdb.set_trace()
        return self.loss

    
    def forward_test(self,pts,data=None):
        """ creates embedding from pts
        Args:
            pts (n*3*12*2): numpy float32 array
        """
        if data is None:
            data = np.asarray([])
            data = data.reshape((pts.shape[0],-1))
        feat = extract_features_from_frames(pts).astype(np.float32)
        featshape = feat.shape
        feat = self.scaler.transform(feat.reshape((-1,60))).reshape(featshape)
        feat = torch.from_numpy(feat.transpose(0,2,1).astype(np.float32)).to(self.device)
        self.pred_test = self.normalize(self.net(feat))
        feat = self.pred_test.detach().cpu().numpy()
        feat = np.nan_to_num(feat)
        return np.hstack((data,feat))

    def set_input(self,data):
        self.data = data
        self.anchor_data = data[0]
        self.pos_data = data[1]
        self.neg_data = data[2]

    def normalize(self,x):
        return F.normalize(x, p=2, dim=-1)

    def forward(self):
        self.pred =self.normalize(self.net(self.anchor_data))
        self.pos_pred = self.normalize(self.net(self.pos_data))
        self.neg_pred = self.normalize(self.net(self.neg_data))

    def train_model(self):
        for epoch in range(50):
            for i, data in enumerate(self.dataloader, 0):
                    self.set_input(data)
                    self.optimizer.zero_grad()
                    self.forward()
                    loss = self.calc_loss()
                    loss.backward()
                    self.optimizer.step()
                    self.writer.add_scalar('training loss',loss.item(),i+epoch * len(self.dataloader))
            print('Epoch %d, loss %.3f'%(epoch,loss.item()))
            #Save model
            os.makedirs(os.path.join(os.getcwd(),'helper/pointnet/checkpoints',self.config['exp_name']),exist_ok=True)
            torch.save(self.net.state_dict(), os.path.join(os.getcwd(),'helper/pointnet/checkpoints',self.config['exp_name'],'latest.pth'))

if __name__ == '__main__':
    config = dict(feature_transform = True,
                  global_feat = True,
                  learning_rate = 0.005,
                  optimizer = 'adam',
                  in_channels = 10,
                  load_path = None,
                  out_features = 15,
                  device = 'cuda:0',
                  exp_name = 'exp0',
                  )
    mousemodel = model(config=config)
    mousemodel.train_model()