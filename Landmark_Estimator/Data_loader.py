import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import numpy as np

from typing import List
import random

class GazeDataset_2Dlandmarks(Dataset):
    '''
    load 2D landmarks template/label along with corresponding gaze info
    '''
    def __init__(self, dataset_path: str,
                keys_to_use: List[str] = None, 
                sub_folder='',
                is_shuffle=True,
                index_file=None, 
                is_load_label=True,
                normalize_landmarks=True,
                device = 'cpu'):
        self.path = dataset_path
        self.hdfs = {}
        self.template_lm = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label
        self.device = device
        self.selected_keys = [k for k in keys_to_use]
        self.normalize_lm = normalize_landmarks
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path,f'processed_{self.selected_keys[num_i]}')
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
        # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
        assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for subjec_id in range(0, len(self.selected_keys)):
                hdfs_file = self.hdfs[subjec_id]
                n = hdfs_file["face_patch"].shape[0]
                self.template_lm[subjec_id] = self._find_template_2dlm(hdfs_file)
                self.idx_to_kv += [(subjec_id, person_id) for person_id in range(n)
                                    if hdfs_file['valid_mask'][person_id] ] #valid frame in subject num_i
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)
        
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.hdf = None
        self.debug(0)

    def _find_template_2dlm(self,hdf_file):
        '''
        find 2d landmark template(where eye gaze looking straight forward) for each subject
        '''
        min_index = np.argmin(np.linalg.norm(hdf_file['face_gaze'],axis=1))
        
        lm2d_temp = hdf_file['lm2d'][min_index]
        lm2d_temp = lm2d_temp.astype('float')
        return lm2d_temp


    def __len__(self):
        return len(self.idx_to_kv)
    
    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self,idx):
        subject_id , person_id = self.idx_to_kv[idx]
        file_path = os.path.join(self.path,f'processed_{self.selected_keys[subject_id]}')
        self.hdf = h5py.File(file_path, 'r', swmr=True)

        gaze_label = self.hdf['face_gaze'][person_id]
        gaze_label = gaze_label.astype('float')
        gaze_tensor = (torch.from_numpy(gaze_label)).to(self.device)

        lm2d = self.hdf['lm2d'][person_id]
        lm2d = lm2d.astype('float')
        lm2d_tensor = (torch.from_numpy(lm2d)).to(self.device)

        lm2d_template = (torch.from_numpy(self.template_lm[subject_id])).to(self.device)

        if self.normalize_lm:
            h,w,_ = self.hdf['face_patch'][person_id].shape
            lm2d_tensor = self._normalize_lm(lm2d_tensor,h,w)
            lm2d_template = self._normalize_lm(lm2d_template,h,w)
            
        data_info ={
            'gaze' : gaze_tensor,
            'lm2d_label' : lm2d_tensor, ##unit in pixel, better nomailzed from -1 to 1
            'lm2d_template' : lm2d_template
        }
        return data_info

    def _normalize_lm(self,lm,h,w):
        lm = lm.view(-1,2)
        lm[:,0] = (lm[:,0]/h - 0.5) * 2
        lm[:,1] = (lm[:,1]/w - 0.5) * 2
        return lm.view(-1)

    def debug(self,idx):
        subject_id , person_id = self.idx_to_kv[idx]
        file_path = os.path.join(self.path,f'processed_{self.selected_keys[subject_id]}')
        self.hdf = h5py.File(file_path, 'r', swmr=True)

        gaze_label = self.hdf['face_gaze'][person_id]
        gaze_label = gaze_label.astype('float')
        gaze_tensor = (torch.from_numpy(gaze_label)).to(self.device)

        lm2d = self.hdf['lm2d'][person_id]
        lm2d = lm2d.astype('float')
        lm2d_tensor = (torch.from_numpy(lm2d)).to(self.device)

        lm2d_template = (torch.from_numpy(self.template_lm[subject_id])).to(self.device)

        if self.normalize_lm:
            h,w,_ = self.hdf['face_patch'][person_id].shape
            import ipdb
            ipdb.set_trace()
            lm2d_tensor = self._normalize_lm(lm2d_tensor,h,w)
            lm2d_template = self._normalize_lm(lm2d_template,h,w)
            
        data_info ={
            'gaze' : gaze_tensor,
            'lm2d_label' : lm2d_tensor, ##unit in pixel, better nomailzed from -1 to 1
            'lm2d_template' : lm2d_template
        }
        return data_info

        

if __name__ == '__main__':
    dataset_config={
    'dataset_path': '../XGaze_data/processed_data/',
    'keys_to_use':['subject0000'], 
    'sub_folder':'train',
    'is_shuffle':True,
    'index_file':None, 
    'is_load_label':True,
    'device': 'cpu',

    }

    dataset = GazeDataset_2Dlandmarks(**dataset_config)
    data_loader = DataLoader(dataset,batch_size=2,num_workers=4,drop_last=True)
    for iter,batch in enumerate(data_loader):
        import ipdb;
        ipdb.set_trace()
        pass
