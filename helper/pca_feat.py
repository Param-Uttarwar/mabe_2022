import sys
import pickle
sys.path.append('/data/param/AIcrowd/mouse/helper')
import numpy as np
import os
import cv2
from os.path import join
from utils import scale_data,fill_holes,normalize_vec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_mean_pca(frame_keyps,data=None):
    """ Dot product between mouse neck-base and other mouses
    mean dist = 268

    Args:
        frame_keyps ((n,3,12,2)): mouse keypoints
        data ((n,m), optional): _description_. Defaults to np.asarray([]).
    """
    if data is None:
        data = np.asarray([])
        data = data.reshape((frame_keyps.shape[0],-1))

    feat = do_PCA(frame_keyps)
    #Do mean
    feat = np.nanmean(feat,axis=1)
    feat = feat.reshape((feat.shape[0],-1))
    feat = np.nan_to_num(feat,nan= 0)
    #PCA_speed 
    feat_speed = np.repeat(np.gradient(feat[::10],axis=0),10,axis=0)
    
    return np.hstack((data,feat))

def normalize_mouse(data,save=False):
    """finds the principal direction and point, rotate all files to convert them to same space
    saves the data in normalized_mouse_poses.npy

    Args:
        data (n*3*12*2): data to normalize
    """
    normalized_mousedata = []
    for i in range(data.shape[0]):
        for mouse in range(data.shape[1]):
            if np.sum(np.isnan(data[i,mouse,:,:])) == 0:
                mouse_center = np.mean(data[i,mouse,[3,6],:],axis=0)
                mouse_pts_norm = data[i,mouse,:,:] - mouse_center[None,:]
                #Calculate angle
                mouse_vec_norm = normalize_vec(data[i,mouse,3,:] - data[i,mouse,6,:])
                x_vec = np.asarray([1,0])
                angle = np.arctan2(mouse_vec_norm[1], mouse_vec_norm[0]) - np.arctan2(x_vec[1], x_vec[0])
                angle = angle*180/np.pi
                rotate_matrix = cv2.getRotationMatrix2D(center=(0,0), angle=angle, scale=1)
                #Rotate
                rotated_pts = np.matmul(rotate_matrix,np.concatenate((mouse_pts_norm,np.ones((12,1))),axis=1).T).T
                normalized_mousedata.append(rotated_pts)
            elif not save:
                normalized_mousedata.append(np.full((12,2),fill_value=np.nan))
    normalized_mousedata = np.asarray(normalized_mousedata)
    if save:
        np.save(join(os.getcwd(),'data','normalized_mouse_poses.npy'),normalized_mousedata)
    normalized_mousedata = normalized_mousedata.reshape((data.shape[0],data.shape[1],12,2))
    return normalized_mousedata

def train_PCA():
    """train PCA from normalized pose data
    """
    data = np.load(os.path.join(os.getcwd(),'data','normalized_mouse_poses.npy'))
    assert len(data.shape) == 3
    #Delete nan
    data =np.delete(data,np.unique(np.where(np.isnan(data)==1)[0]),axis=0)
    data = data.reshape((data.shape[0],-1))
    scaled_data, scaler = scale_data(data,return_scaler=True)
    #Save scaler
    pickle.dump(scaler,open(join(os.getcwd(),'data','PCAscaler.pkl'),'wb'))
    #Do PCA
    pca = PCA(n_components=10)
    pca.fit(scaled_data)
    pickle.dump(pca,open(join(os.getcwd(),'data','PCAtrans.pkl'),'wb'))
    
def do_PCA(data):
    """scaling and doing PCA

    Args:
        data (n*11*24*2) or n*19*2 : _description_
        train (bool, optional): _description_. Defaults to False.
    """
    datashape = data.shape
    #load stuff
    pca = pickle.load(open(join(os.getcwd(),'data','PCAtrans.pkl'),'rb'))
    scaler = pickle.load(open(join(os.getcwd(),'data','PCAscaler.pkl'),'rb'))
    normalize_pose = normalize_mouse(data) #n*11*12*2
    normalize_pose = normalize_pose.reshape((-1,24)) #11n*38
    pca_pose = np.full((normalize_pose.shape[0],pca.n_components),np.nan)
    nan_index = np.isnan(normalize_pose[:,0])
    try:
        normalize_pose = scaler.transform(normalize_pose[~nan_index])
        normalize_pose = pca.transform(normalize_pose)
        pca_pose[~nan_index] = normalize_pose
        pca_pose = pca_pose.reshape((*data.shape[:2],-1))
    except:
        pca_pose = pca_pose.reshape((*data.shape[:2],-1))
        return pca_pose
    return pca_pose    

if __name__ ==  '__main__':
    dataset = np.load(os.path.join(os.getcwd(),'data','user_train.npy'),allow_pickle=True).item()
    seq = dataset['sequences']
    #Convert all data into one array
    data = [seq[key]['keypoints'] for key in seq.keys()]
    data = np.asarray(data)
    labels = [seq[key]['annotations'] for key in seq.keys()]
    labels = np.asarray(labels)
    all_data = data.reshape((-1,*data.shape[2:]))
    #Normalize data
    norm_data = normalize_mouse(all_data,save=True)
    #Train data
    train_PCA()