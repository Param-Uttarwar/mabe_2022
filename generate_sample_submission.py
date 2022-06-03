import numpy as np
import os
import cv2
from helper.features import *
import itertools

from helper.utils import fill_holes,scale_data,check_nan
from helper.pca_feat import get_mean_pca


if __name__ == '__main__':

    subm_data = np.load(os.path.join(os.getcwd(),'data','submission_data.npy'),allow_pickle=True).item()
    frame_map = np.load(os.path.join(os.getcwd(),'data','frame_number_map.npy'),allow_pickle=True).item()
    outdir = os.path.join(os.getcwd(),'results','submissions')
    os.makedirs(outdir, exist_ok=True)
    feature_dict = {0:get_inter_mouse_dist,1:get_inter_mouse_angles,2:get_mouse_cornerdist,3:get_inter_mouse_angles2,\
                    8: get_mouse_nosetail_dist, 9:get_mouse_nosenose_dist, 10:get_mouse_speeds2, \
                    11:get_headbody_angle, 12: get_mouse_nosespeed, 13: get_mean_pca} #| # Velocity vector, 
    feat_list = [0,1,2,3,8,9,10,11,12,13]
    #Generate data
    clean_data = []
    data_dict = {}
    for key in subm_data['sequences'].keys():
        print(key)
        data = None
        seq_data = subm_data['sequences'][key]['keypoints']
        #Clean data
        cleaned_seq_data = fill_holes(seq_data)
        if cleaned_seq_data.size == 0:
            cleaned_seq_data = seq_data
        clean_data.append(cleaned_seq_data)
        #Calcualte all features
        for feat in feat_list:
            data = feature_dict[feat](cleaned_seq_data,data)
        data_dims = data.shape[1]
        data_dict[key] = data
        import pdb;pdb.set_trace()

    #Generate submission
    num_total_frames = np.sum([seq["keypoints"].shape[0] for _, seq in subm_data['sequences'].items()])
    embeddings_array = np.empty((num_total_frames, data_dims), dtype=np.float32)
    frame_number_map = {}
    start = 0
    for sequence_key in subm_data['sequences']:
        end = start + 1800
        embeddings = data_dict[sequence_key]
        embeddings_array[start:end] = embeddings
        frame_number_map[sequence_key] = (start, end)
        start = end

    assert end == num_total_frames     
    embeddings_array = scale_data(embeddings_array)
    submission_dict = {"frame_number_map": frame_number_map, "embeddings": embeddings_array}
    np.save(os.path.join(outdir,'submission.npy'),submission_dict,allow_pickle=True)


    



    