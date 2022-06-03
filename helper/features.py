
import numpy as np
import os
import cv2
from os.path import join
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from helper.utils import fill_holes
IMAGE_SIZE = 850

def moving_average(arr, n=10) :
    axis = 0
    pad_arr = [(n//2-1,n//2)] + [(0,0)]*(len(arr.shape)-1)
    arr = np.pad(arr,pad_arr,mode = 'reflect')
    ret = np.cumsum(arr,axis=axis, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_inter_mouse_dist(frame_keyps,data=None):
    """Calculates distances between each mouse and the other mouses

    Args:
        frame_keyps ((n,3,12,2)): mouse keypoints
        data ((n,m), optional): _description_. Defaults to np.asarray([]).
    """
    if data is None:
        data = np.asarray([])
        data = data.reshape((frame_keyps.shape[0],-1))
    #Calculate distance    
    mouse_pos = np.mean(frame_keyps[:,:,[3,6],:],axis=2)
    pos1, pos2 = mouse_pos[:,[0,1,2],:],mouse_pos[:,[1,2,0],:]
    mouse_dists = np.linalg.norm(pos1-pos2,axis=2)
    return np.hstack((data,mouse_dists))

def get_inter_mouse_angles(frame_keyps,data=None):
    """ Angle between mouse neck-base and other mouses
    mean dist = 268

    Args:
        frame_keyps ((n,3,12,2)): mouse keypoints
        data ((n,m), optional): _description_. Defaults to np.asarray([]).
    """
    if data is None:
        data = np.asarray([])
        data = data.reshape((frame_keyps.shape[0],-1))
    #Calulate vectors
    mouse_keyps = frame_keyps[:,:,[3,6],:].astype(float)
    mouse_vecs = mouse_keyps[:,:,0] - mouse_keyps[:,:,1]
    mouse_vecs_norm = mouse_vecs/np.linalg.norm(mouse_vecs,axis=2)[:,:,None]
    mouse_vecs_norm = np.repeat(mouse_vecs_norm,2,axis=1) # repeat 2x 
    #Calculate angles
    mouse_pos = np.mean(frame_keyps[:,:,[3,6],:],axis=2)
    mouse_pos_vec = mouse_pos[:,[0,0,1,1,2,2]]-mouse_pos[:,[1,2,0,2,0,1]]
    mouse_pos_vec_norm = mouse_pos_vec/np.linalg.norm(mouse_pos_vec,axis=2)[:,:,None]
    dot_product = np.sum(np.multiply(mouse_vecs_norm,mouse_pos_vec_norm),axis=2)
    mouse_angles = np.arccos(dot_product)
    #Replace nan with Pi/2
    mouse_angles = np.nan_to_num(mouse_angles,nan= np.pi/2)
    return np.hstack((data,mouse_angles))

def get_mouse_cornerdist(frame_keyps,data=None):
    """calcuates how far from wall 
    mean dist = 425

    Args:
        frame_keyps ((n,3,12,2)): mouse keypoints
        data ((n,m), optional): _description_. Defaults to np.asarray([]).
    """
    if data is None:
        data = np.asarray([])
        data = data.reshape((frame_keyps.shape[0],-1))
    #Calculate distance
    mouse_pos = np.mean(frame_keyps[:,:,[3,6],:],axis=2)
    mouse_dist = np.minimum(mouse_pos,IMAGE_SIZE - mouse_pos)
    mouse_dist = mouse_dist.reshape((mouse_dist.shape[0],-1))
    return np.hstack((data,mouse_dist))    

def get_inter_mouse_angles2(frame_keyps,data=None):
    """ Angle between mouse nose-neck and other mouses
    mean dist = 268

    Args:
        frame_keyps ((n,3,12,2)): mouse keypoints
        data ((n,m), optional): _description_. Defaults to np.asarray([]).
    """
    if data is None:
        data = np.asarray([])
        data = data.reshape((frame_keyps.shape[0],-1))
    #Calulate vectors
    mouse_keyps = frame_keyps[:,:,[0,3],:].astype(float)
    mouse_vecs = mouse_keyps[:,:,0] - mouse_keyps[:,:,1]
    mouse_vecs_norm = mouse_vecs/np.linalg.norm(mouse_vecs,axis=2)[:,:,None]
    mouse_vecs_norm = np.repeat(mouse_vecs_norm,2,axis=1) # repeat 2x 
    #Calulate angles
    mouse_pos = np.mean(frame_keyps[:,:,[3,6],:],axis=2)
    mouse_pos_vec = mouse_pos[:,[0,0,1,1,2,2]]-mouse_pos[:,[1,2,0,2,0,1]]
    mouse_pos_vec_norm = mouse_pos_vec/np.linalg.norm(mouse_pos_vec,axis=2)[:,:,None]
    dot_product = np.sum(np.multiply(mouse_vecs_norm,mouse_pos_vec_norm),axis=2)
    mouse_angles = np.arccos(dot_product)
    #Replace nan with Pi/2
    mouse_angles = np.nan_to_num(mouse_angles,nan= np.pi/2)
    return np.hstack((data,mouse_angles))

def get_mouse_nosetail_dist(frame_keyps,data=None):
    """Nose-tail distance of mouse with other mouses

    Args:
        frame_keyps ((n,3,12,2)): mouse keypoints
        data ((n,m), optional): _description_. Defaults to None.
    """
    if data is None:
        data = np.asarray([])
        data = data.reshape((frame_keyps.shape[0],-1))
    #Calculate distance
    mouse_keyps = frame_keyps.astype(float)
    mouse_dist = mouse_keyps[:,[0,0,1,1,2,2],0,:]-mouse_keyps[:,[1,2,0,2,0,1],9,:] #0 : nose, #9 : tail base
    mouse_dist = np.linalg.norm(mouse_dist,axis=-1) 
    mouse_dist = np.nan_to_num(mouse_dist,nan=1000)
    #Sort by distance
    for i in range(len(mouse_dist)):
        mouse_dist[i].sort()
    return np.hstack((data,mouse_dist[:,:2]))

def get_mouse_nosenose_dist(frame_keyps,data=None):
    """Nose-nose distance of mouse with other mouses

    Args:
        frame_keyps ((n,3,12,2)): mouse keypoints
        data ((n,m), optional): _description_. Defaults to None.
    """
    if data is None:
        data = np.asarray([])
        data = data.reshape((frame_keyps.shape[0],-1))
    #Calculate distance
    mouse_keyps = frame_keyps.astype(float)
    mouse_dist = mouse_keyps[:,[0,0,1,1,2,2],0,:]-mouse_keyps[:,[1,2,0,2,0,1],0,:] #0 : nose, #9 : tail base
    mouse_dist = np.linalg.norm(mouse_dist,axis=-1) 
    mouse_dist = np.nan_to_num(mouse_dist,nan=1000)
    #Sort by distance
    for i in range(len(mouse_dist)):
        mouse_dist[i].sort()
    return np.hstack((data,mouse_dist[:,:2]))

def get_mouse_speeds2(frame_keyps,data=None):
    """frame_keyps ((n,3,12,2)): mouse keypoints

    Args:
        frame_keyps (_type_): _description_
        data (_type_, optional): _description_. Defaults to None.
    """
    centroid = moving_average(np.mean(frame_keyps,axis=2))
    #Calcluate speed of centroid
    grad = np.gradient(centroid,axis=0)
    speedcent = np.linalg.norm(grad,axis=-1).squeeze()
    speedcent = np.nan_to_num(speedcent,nan=0)
    #Sort by speed
    for i in range(len(speedcent)):
        speedcent[i].sort()
    return np.hstack((data,speedcent))

def get_headbody_angle(frame_keyps,data=None):
    """Caculate head to body angle of mouse with other mouses

        Args:
            frame_keyps ((n,3,12,2)): mouse keypoints
            data ((n,m), optional): _description_. Defaults to None.
    """
    if data is None:
        data = np.asarray([])
        data = data.reshape((frame_keyps.shape[0],-1))
    #Calculate vectors
    mouse_keyps = frame_keyps[:,:,[0,3],:].astype(float)
    mouse_vecs = mouse_keyps[:,:,0] - mouse_keyps[:,:,1]
    mouse_vecs_norm1 = mouse_vecs/np.linalg.norm(mouse_vecs,axis=2)[:,:,None]
    #Calculate angles
    mouse_keyps = frame_keyps[:,:,[3,9],:].astype(float)
    mouse_vecs = mouse_keyps[:,:,0] - mouse_keyps[:,:,1]
    mouse_vecs_norm2 = mouse_vecs/np.linalg.norm(mouse_vecs,axis=2)[:,:,None]
    dot_product = np.sum(np.multiply(mouse_vecs_norm1,mouse_vecs_norm2),axis=2)
    mouse_angles = np.arccos(dot_product)
    #Replace nan with Pi/2
    mouse_angles = np.nan_to_num(mouse_angles,nan=0)
    return np.hstack((data,mouse_angles))

def get_mouse_nosespeed(frame_keyps,data=None):
    """Relative speed of nose of the mouse

    Args:
        frame_keyps (_type_): _description_
        data (_type_, optional): _description_. Defaults to None.
    """
    #Calculate speed of nose relative to centroid
    centroid =(np.mean(frame_keyps,axis=2))
    nose_cent = moving_average((frame_keyps[:,:,0,:]-centroid))
    grad = np.gradient(centroid,axis=0)
    speed = np.linalg.norm(grad,axis=-1).squeeze()
    speed = np.nan_to_num(speed,nan=1000)
    #Sort by speed
    for i in range(len(speed)):
        speed[i].sort()
    return np.hstack((data,speed))