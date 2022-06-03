from configparser import NoSectionError
import re
import numpy as np
import cv2
import pickle
import os
from os.path import join
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import itertools
sys.path.append(os.getcwd())
from helper.utils import moving_average,check_zeros,check_nan,scale_data,normalize_vec,fill_holes


def extract_features_from_pairs(frame_keyps,all_perm = False):
    """Extract mouse features from keypoints

    Args:
        frame_keyps ((1800,6,2,12,2)): mouse keypoints (1800 is frames, 6 is permutations, 2 is mouses, 12 is keypoints)
        all_perm (bool) : if saving , take all mouse permutations and downsample by 5 , for training data PCA
    Returns
        feat: (m,6,10) : mouse features (m is frames, 6 is permutations, 10 is features)) 
    """
    #speed
    displacement = np.gradient(moving_average(frame_keyps[...,3,:],n=10)[::5,...],axis=0)
    speeds = np.repeat(np.linalg.norm(displacement,axis=-1),5,0)
    #relative speed
    displacement = np.gradient(moving_average(frame_keyps[...,1,3,:]-frame_keyps[...,0,3,:],n=10)[::5,...],axis=0)
    rel_speed = np.repeat(np.linalg.norm(displacement,axis=-1),5,0)
    #Distances
    nose_tail_dist = np.linalg.norm(frame_keyps[...,0,9,:] - frame_keyps[...,1,0,:],axis=-1)
    neck_neck_dist = np.linalg.norm(frame_keyps[...,0,3,:] - frame_keyps[...,1,3,:],axis=-1)
    dist_from_center = np.linalg.norm(frame_keyps[...,3,:]-425,axis=-1)
    #Angle
    vec0 = normalize_vec(frame_keyps[...,0,6,:] - frame_keyps[...,0,9,:])
    vec1 = normalize_vec(frame_keyps[...,1,3,:] - frame_keyps[...,0,6,:])
    angle_between = np.sum(np.multiply(vec0,vec1),axis=-1)
    #angular_speed
    angle0 = normalize_vec(frame_keyps[...,0,6,:] - frame_keyps[...,0,9,:])
    angle0 = np.arctan2(angle0[...,1],angle0[...,0])
    angle0_speed = np.gradient(moving_average(angle0,n=10)[::5,...],axis=0)
    angle0_speed = np.repeat(np.abs(angle0_speed),5,0)

    angle1 = normalize_vec(frame_keyps[...,1,6,:] - frame_keyps[...,1,9,:])
    angle1 = np.arctan2(angle1[...,1],angle1[...,0])
    angle1_speed = np.gradient(moving_average(angle1,n=10)[::5,...],axis=0)
    angle1_speed = np.repeat(np.abs(angle1_speed),5,0)
    feat = np.concatenate((angle_between[...,None],dist_from_center,neck_neck_dist[...,None],\
                           nose_tail_dist[...,None],speeds,angle0_speed[...,None]\
                            ,angle1_speed[...,None],rel_speed[...,None]),axis=-1)

    return feat


def extract_features_from_frames(frame_keyps, all_perm=False):
    """Extract 6*10 features from keypoints

    Args:
        frame_keyps ((1800,3,12,2)): mouse keypoints (1800 is frames)
    Returns:
        feat: (1800,6,10) : mouse features (1800 is frames, 6 is permutations, 10 is features))
    """
    frame_keyps = [frame_keyps[:,pair,:,:] for pair in itertools.permutations(range(3),2)]
    frame_keyps = np.asarray(frame_keyps).transpose(1,0,2,3,4)
    feat = extract_features_from_pairs(frame_keyps) #1800*6*10

    if all_perm == True:
        #downsample by 10
        feat = feat[::10,:,:]
        #all permutations of 6, downsample by 50 , for training data PCA
        feat = [feat[:,perm,:] for perm in list(itertools.permutations(range(6),6))[::50]]
        feat = np.concatenate((feat),axis=0)
    return feat

def create_dataset():
    """
    Create dataset for training
    """
    dataset = np.load(os.path.join(os.getcwd(),'data','user_train.npy'),allow_pickle=True).item()
    seq = dataset['sequences']
    all_feat = []
    for i,key in enumerate(seq.keys()):
        print('%f Done'%(i/len(seq.keys())))
        data = fill_holes(seq[key]['keypoints'])
        if data.shape[0] < 1800:
            continue
        feat = extract_features_from_frames(data,all_perm=True)
        all_feat.append(feat)
    all_feat = np.asarray((all_feat)) #seq*frame*6*10
    np.save(os.path.join(os.getcwd(),'data','pointnet','dataset.npy'),all_feat)

def train_PCA():
    """Train PCA on dataset features extracted from mouse keypoints
    """
    data = np.load(os.path.join(os.getcwd(),'data','pointnet','dataset.npy'))
    data = data.reshape(-1,6*10)
    scaled_data, scaler = scale_data(data,return_scaler=True)
    #Save scaler
    pickle.dump(scaler,open(join(os.path.join(os.getcwd(),'data','pointnet'),'scaler.pkl'),'wb'))
    #Train PCA
    pca = PCA(n_components=25)
    pca.fit(scaled_data)
    pickle.dump(pca,open(join(os.path.join(os.getcwd(),'data','pointnet'),'PCAtrans.pkl'),'wb'))
    pickle.dump(pca.explained_variance_ratio_,open(join(os.path.join(os.getcwd(),'data','pointnet'),'PCAweights.pkl'),'wb'))    


class mousePCA():
    def __init__(self) -> None:
        self.pca = pickle.load(open(os.path.join(os.getcwd(),'data','pointnet','PCAtrans.pkl'),'rb'))
        self.scaler = pickle.load(open(os.path.join(os.getcwd(),'data','pointnet','scaler.pkl'),'rb'))
        self.weights = pickle.load(open(os.path.join(os.getcwd(),'data','pointnet','PCAweights.pkl'),'rb'))

    def arrange_by_PCA(self,feat):
        """Calculate and returns min. distance of permutations in PCA space 

        Parameters
        ----------
        feat : n*6*10
            batch of keypoints  

        Returns:
        ----------
        min_dist : n
            minimum distance from anchor point in PCA space
        """
        anchor = feat[0] # select anchor point for getting positive and negative examples (batch size default is 512)
        anchor = self.pca.transform(anchor.reshape(-1,6*10))
        #Permute all
        feat_all = np.asarray([feat[:,perm,:] for perm in list(itertools.permutations(range(6),6))])
        pca_all = self.pca.transform(feat_all.reshape(-1,6*10))
        pca_all = pca_all.reshape((*feat_all.shape[:2],pca_all.shape[-1]))
        #Calculate wieghted L2 distance of each permuation in PCA space
        dist = np.linalg.norm(self.weights[None,None,:]*(pca_all-anchor[None,...]),axis=-1)
        #Select minimum distance for each sample
        min_dists = np.min(dist,axis=0)
        return min_dists

if __name__=='__main__':
    create_dataset()
    train_PCA()