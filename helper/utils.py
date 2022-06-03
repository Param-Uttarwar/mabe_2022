from sklearn import preprocessing
import numpy as np
import copy
import os


def moving_average(arr, n=30) :
    axis = 0
    pads = [(n//2-1,n//2)]
    for i in range(len(arr.shape)-1):
        pads.append((0,0))
    arr = np.pad(arr,pads,mode = 'reflect')
    ret = np.cumsum(arr,axis=axis, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def normalize_vec(v):
    """_summary_

    Args:
        v (any shape): last dimension should is normalized across

    Returns:
        _type_: _description_
    """
    return v/(np.linalg.norm(v,axis=-1)[...,None]+0.001)
    
def check_nan(data):
    return np.count_nonzero(np.isnan(data))/data.size
    

def check_zeros(data):
    return 1 - np.count_nonzero(data)/data.size

def scale_data(data,return_scaler=False):
    scaler = preprocessing.StandardScaler().fit(data)
    scaled_data = scaler.transform(data)

    if return_scaler:
        return scaled_data, scaler

    return scaled_data

def validate_submission(submission, submission_clips):

    if not isinstance(submission, dict):
      print("Submission should be dict")
      return False

    if 'frame_number_map' not in submission:
      print("Frame number map missing")
      return False

    if 'embeddings' not in submission:
        print('Embeddings array missing')
        return False
    elif not isinstance(submission['embeddings'], np.ndarray):
        print("Embeddings should be a numpy array")
        return False
    elif not len(submission['embeddings'].shape) == 2:
        print("Embeddings should be 2D array")
        return False
    elif not submission['embeddings'].shape[1] <= 100:
        print("Embeddings too large, max allowed is 100")
        return False
    elif not isinstance(submission['embeddings'][0, 0], np.float32):
        print(f"Embeddings are not float32")
        return False

    
    total_clip_length = 0
    for key in submission_clips['sequences']:
        start, end = submission['frame_number_map'][key]
        clip_length = submission_clips['sequences'][key]['keypoints'].shape[0]
        total_clip_length += clip_length
        if not end-start == clip_length:
            print(f"Frame number map for clip {key} doesn't match clip length")
            return False
            
    if not len(submission['embeddings']) == total_clip_length:
        print(f"Emebddings length doesn't match submission clips total length")
        return False

    if not np.isfinite(submission['embeddings']).all():
        print(f"Emebddings contains NaN or infinity")
        return False

    print("All checks passed")
    return True


def fill_holes(data):
    clean_data = copy.deepcopy(data)
    for m in range(3):
        holes = np.where(clean_data[0,m,:,0]==0)
        if not holes:
            continue
        for h in holes[0]:
            sub = np.where(clean_data[:,m,h,0]!=0)
            if(sub and sub[0].size > 0):
                clean_data[0,m,h,:] = clean_data[sub[0][0],m,h,:]
            else:
              return np.empty((0))
    
    for fr in range(1,np.shape(clean_data)[0]):
        for m in range(3):
            holes = np.where(clean_data[fr,m,:,0]==0)
            if not holes:
                continue
            for h in holes[0]:
                clean_data[fr,m,h,:] = clean_data[fr-1,m,h,:]
    return clean_data


def clean_train_data():
    """Reads, cleans and saves data for further processing 
    """
    train_data = np.load(os.path.join(os.getcwd(),'data','user_train.npy'),allow_pickle=True).item()

    clean_data_dict = {'sequences':{}}
    counts = {'pos' : 0, 'cleaned' : 0, 'neg' : [], }
    for sequence_key in train_data['sequences']:
        anno = train_data['sequences'][sequence_key]['annotations']
        data = train_data['sequences'][sequence_key]['keypoints']
        if not np.count_nonzero(data) == data.size:
            clean_data = fill_holes(data)
            if not np.count_nonzero(clean_data) == data.size:
                counts['neg'].append(clean_data)
            else:
                counts['cleaned']+=1
                clean_data_dict['sequences'][sequence_key]={'keypoints':clean_data,'annotations':anno}
        else:
            counts['pos']+=1
            clean_data_dict['sequences'][sequence_key]={'keypoints':data,'annotations':anno}

    print(counts)
    np.save(os.path.join(os.getcwd(),'data','user_train_cleaned.npy'),clean_data_dict,allow_pickle=True)



if __name__ == '__main__':

    train_data = np.load(os.path.join(os.getcwd(),'data','user_train_cleaned.npy'),allow_pickle=True).item()

    counts = {'pos' : 0, 'cleaned' : 0, 'neg' : [], }
    for sequence_key in train_data['sequences']:
        anno = train_data['sequences'][sequence_key]['annotations']
        data = train_data['sequences'][sequence_key]['keypoints']
        if not np.count_nonzero(data) == data.size:
            clean_data = fill_holes(data)
            if not np.count_nonzero(clean_data) == data.size:
                counts['neg'].append(clean_data)
            else:
                counts['cleaned']+=1        
        else:
            counts['pos']+=1

    print(counts)