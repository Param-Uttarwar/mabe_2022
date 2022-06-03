import torch
import torch.nn as nn
import os
from os.path import join
import sys
import numpy as np
sys.path.append(os.getcwd())
from helper.utils import fill_holes,scale_data
import argparse
from helper.pointnet.train import model

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

config = dict(feature_transform = True,
                  global_feat = True,
                  learning_rate = 0.005,
                  optimizer = 'adam',
                  in_channels = 10,
                  out_features = 15,
                  load_path = join(os.getcwd(),'helper/pointnet/checkpoints/exp0/latest.pth'),
                  device = 'cuda:0',
                  )
pointnet_model = model(config)
pointnet_model.eval()


if __name__ == '__main__':
    subm_data = np.load(os.path.join(os.getcwd(),'data','submission_data.npy'),allow_pickle=True).item()
    frame_map = np.load(os.path.join(os.getcwd(),'data','frame_number_map.npy'),allow_pickle=True).item()

    data_dict = {}
    for key in subm_data['sequences'].keys():
        data = None
        seq_data = subm_data['sequences'][key]['keypoints']
        cleaned_seq_data = fill_holes(seq_data)
        if cleaned_seq_data.size == 0:
            cleaned_seq_data = seq_data
        data = pointnet_model.forward_test(cleaned_seq_data)
        data_dims = data.shape[1]
        data_dict[key] = data
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
    embeddings_array = scale_data(embeddings_array)
    np.save(join(os.getcwd(),'results','intermediate_results','pointnet_feat.npy'),embeddings_array)