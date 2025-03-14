from cmath import isnan
import select


import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from lip_nopre_helpers_deform_yuan import *

import os
import numpy as np
import pickle as pkl
import shutil

from HeadNeRFOptions import BaseOptions
from NetWorks.HeadNeRFNet import HeadNeRFNet
from Utils.HeadNeRFLossUtils import HeadNeRFLossUtils
from Utils.RenderUtils import RenderUtils
from Utils.Eval_utils import calc_eval_metrics
from tqdm import tqdm
import cv2
from Utils.Log_utils import log
from Utils.D6_rotation import gaze_to_d6

import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
#from CodeTalker.models.stage1_vocaset import VQAutoEncoder as q_Model
#from CodeTalker.base.baseTrainer import load_state_dict as load_state_dict_


from cmath import isnan
import select


import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from lip_nopre_helpers_deform_yuan import *

import os
import numpy as np

import shutil

from HeadNeRFOptions import BaseOptions
from NetWorks.HeadNeRFNet import HeadNeRFNet
from Utils.HeadNeRFLossUtils import HeadNeRFLossUtils
from Utils.RenderUtils import RenderUtils
from Utils.Eval_utils import calc_eval_metrics
from tqdm import tqdm
import cv2
from Utils.Log_utils import log
from Utils.D6_rotation import gaze_to_d6
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


from s_utils.croper import Preprocesser
from s_utils.init_path import init_path
from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser
from s_test_audio2coeff import Audio2Coeff
import safetensors
import safetensors.torch 
import torch.optim as optim
from s_audio2pose_models.audio2pose import Audio2Pose
from s_audio2exp_models.networks import SimpleWrapperV2 
from s_audio2exp_models.audio2exp import Audio2Exp
from s_utils.safetensor_helper import load_x_from_safetensor  
from DataProcess.genlm_loss import Gen2DLandmarks
from lipsrc.lipread_loss import Trainer as lipTrainer
from lipsrc.lipspectre import SPECTRE
from lipconfig import parse_args





#wav2lip
from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, wav_audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from wav_models import Wav2Lip
import platform



# sadtalker
import numpy as np
import cv2, os, sys, torch
from tqdm import tqdm
from PIL import Image 

# 3dmm extraction
import safetensors
import safetensors.torch 
from s_face3d.util.preprocess import align_img
from s_face3d.util.load_mats import load_lm3d
from s_face3d.models import networks

from scipy.io import loadmat, savemat
from s_utils.croper import Preprocesser


import warnings
from s_utils.safetensor_helper import load_x_from_safetensor 
warnings.filterwarnings("ignore")
from s_utils.init_path import init_path
#gpu_list = str([2])
#os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
#print('export CUDA_VISIBLE_DEVICES=' + gpu_list)




#wav2lip
def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()






def split_coeff(coeffs):
        """
        Return:
            coeffs_dict     -- a dict of torch.tensors

        Parameters:
            coeffs          -- torch.tensor, size (B, 256)
        """
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }


class CropAndExtract():
    def __init__(self, sadtalker_path, device):

        self.propress = Preprocesser(device)
        self.net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='').to(device)
        
        if sadtalker_path['use_safetensor']:
            checkpoint = safetensors.torch.load_file(sadtalker_path['checkpoint'])    
            self.net_recon.load_state_dict(load_x_from_safetensor(checkpoint, 'face_3drecon'))
        else:
            checkpoint = torch.load(sadtalker_path['path_of_net_recon_model'], map_location=torch.device(device))    
            self.net_recon.load_state_dict(checkpoint['net_recon'])

        self.net_recon.eval()
        self.lm3d_std = load_lm3d(sadtalker_path['dir_of_BFM_fitting'])
        self.device = device
    
    def generate(self, input_path, save_dir, crop_or_resize='crop', source_image_flag=False, pic_size=256):

        pic_name = os.path.splitext(os.path.split(input_path)[-1])[0]  

        landmarks_path =  os.path.join(save_dir, pic_name+'_landmarks.txt') 
        #coeff_path =  os.path.join(save_dir, pic_name+'.mat')  
        png_path =  os.path.join(save_dir, pic_name+'.png')  

        #load input
        if not os.path.isfile(input_path):
            raise ValueError('input_path must be a valid path to video/image file')
        elif input_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            # loader for first frame
            full_frames = [cv2.imread(input_path)]
            fps = 25
        else:
            # loader for videos
            video_stream = cv2.VideoCapture(input_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            full_frames = [] 
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break 
                full_frames.append(frame) 
                if source_image_flag:
                    break

        x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames] 

        #### crop images as the 
        if 'crop' in crop_or_resize.lower(): # default crop
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        elif 'full' in crop_or_resize.lower():
            x_full_frames, crop, quad = self.propress.crop(x_full_frames, still=True if 'ext' in crop_or_resize.lower() else False, xsize=512)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
            crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)
        else: # resize mode
            oy1, oy2, ox1, ox2 = 0, x_full_frames[0].shape[0], 0, x_full_frames[0].shape[1] 
            crop_info = ((ox2 - ox1, oy2 - oy1), None, None)

        frames_pil = [Image.fromarray(cv2.resize(frame,(pic_size, pic_size))) for frame in x_full_frames]
        if len(frames_pil) == 0:
            print('No face is detected in the input file')
            return None, None

        # save crop info
        for frame in frames_pil:
            cv2.imwrite(png_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

        # 2. get the landmark according to the detected face. 
        if not os.path.isfile(landmarks_path): 
            lm = self.propress.predictor.extract_keypoint(frames_pil, landmarks_path)
        else:
            print(' Using saved landmarks.')
            lm = np.loadtxt(landmarks_path).astype(np.float32)
            lm = lm.reshape([len(x_full_frames), -1, 2])

        #if not os.path.isfile(coeff_path):
            # load 3dmm paramter generator from Deep3DFaceRecon_pytorch 
        video_coeffs, full_coeffs = [],  []
        #print(len(frames_pil))
        for idx in range(len(frames_pil)):
            frame = frames_pil[idx]
            W,H = frame.size
            lm1 = lm[idx].reshape([-1, 2])
        
            if np.mean(lm1) == -1:
                lm1 = (self.lm3d_std[:, :2]+1)/2.
                lm1 = np.concatenate(
                    [lm1[:, :1]*W, lm1[:, 1:2]*H], 1
                )
            else:
                lm1[:, -1] = H - 1 - lm1[:, -1]

            trans_params, im1, lm1, _ = align_img(frame, lm1, self.lm3d_std)

            trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
            im_t = torch.tensor(np.array(im1)/255., dtype=torch.float32).permute(2, 0, 1).to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                full_coeff = self.net_recon(im_t)
                coeffs = split_coeff(full_coeff)

            pred_coeff = {key:coeffs[key].cpu().numpy() for key in coeffs}
            return pred_coeff['exp']

            pred_coeff = np.concatenate([
                pred_coeff['exp'], 
                pred_coeff['angle'],
                pred_coeff['trans'],
                trans_params[2:][None],
                ], 1)
            video_coeffs.append(pred_coeff)
            full_coeffs.append(full_coeff.cpu().numpy())

        
        semantic_npy = np.array(video_coeffs)[:,0] 

            #savemat(coeff_path, {'coeff_3dmm': semantic_npy, 'full_3dmm': np.array(full_coeffs)[0]})

        #return coeff_path, png_path, crop_info













def l1_loss(x, y):
    #return torch.mean((x - y) ** 2)
    return torch.mean(torch.abs(x - y))

def landmark_loss(predict_lm, gt_lm, weight=None):
    """
    weighted mse loss
    Parameters:
        predict_lm    --torch.tensor (B, 68, 2)
        gt_lm         --torch.tensor (B, 68, 2)
        weight        --numpy.array (1, 68)
    """
    if not weight:
        weight = np.ones([68])
        weight[48:68] = 60
        weight = np.expand_dims(weight, 0)
        #weight = torch.tensor(weight).to("cuda")
    loss = torch.sum((predict_lm - gt_lm)**2, dim=-1) * weight
    loss = torch.sum(loss) / (predict_lm.shape[0] * predict_lm.shape[1])
    return loss
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    # elif classname.find('GRU') != -1 or classname.find('LSTM') != -1:
    #     m.weight.data.normal_(0.0, 0.02)
    #     m.bias.data.fill_(0.01)
    else:
        print(classname)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class BasicBlock_AU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(1,1), stride=(1,1),maxpool=(2,2), padding=0):
        super(BasicBlock_AU, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.bn1(out)
        out = self.relu(out)
        return out
        
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), nn.InstanceNorm2d(dim), nn.ReLU(inplace=True)]
        conv_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), nn.InstanceNorm2d(dim)]
        self.conv_blocks = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_blocks(x)
        return out

class RNNModel(nn.Module):
    def __init__(self, input_size=256, hidden_size=256, num_layers=2,batch_first=True,bidirectional=True):
        super(RNNModel, self).__init__()
        #self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = num_layers
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers,
                           batch_first=True,
                           bidirectional=True)
        if bidirectional:
            self.fc1 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, inputs):

        output,_ = self.rnn(inputs)
        return output


class Audio2style(nn.Module):
    def __init__(self, hidden_size=128):
        super(Audio2style, self).__init__()
        '''
        #self.conv_layer_exp = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        #self.pool_layer_exp = nn.MaxPool1d(kernel_size=3, stride=2)
        #self.linear_layer_exp = nn.Linear(48, 29)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=6, stride=1, padding=1)
        #self.conv_layer_exp = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=3)
    
        # the input map is 1 x 12 x 28       
        self.block1 = BasicBlock_AU(2, 8, (1,3), stride=(1,2),maxpool=(1,1)) # 3 x 12 x 13
        self.block2 = BasicBlock_AU(8, 32, kernel_size=(1,3), stride=(1,2),maxpool=(1,1)) # 8 x 12 x 6
        self.block3 = BasicBlock_AU(32, 64, kernel_size=(1,3), stride=(1,2),maxpool=(1,2)) # 16 x 12 x 1
        self.block4 = BasicBlock_AU(64, 128, kernel_size=(3,1), stride=(2,1),maxpool=(1,1)) # 32 x 5 x 1
        self.block5 = BasicBlock_AU(128, 256, kernel_size=(3,1), stride=(2,1),maxpool=(2,1)) # 32 x 1 x 1 
        self.rnn = RNNModel(256, hidden_size)
        self.fc1 = nn.Sequential(nn.Linear(256,128), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5)) # 128
        self.fc2 = nn.Sequential(nn.Linear(128,127), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5)) # 128
        '''
        self.flatten = nn.Flatten()
        self.rnn = RNNModel(80 * 16, 40 * 16)
        self.linear1 = nn.Sequential(nn.Linear(80 * 16, 40 * 16), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        self.linear2 = nn.Sequential(nn.Linear(40 * 16, 20*16), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        self.linear3 = nn.Sequential(nn.Linear(20 * 16, 64), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        #self.linear4 = nn.Sequential(nn.Linear(79*2, 79), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        #self.linear5 = nn.Sequential(nn.Linear(79, 30), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
              
    def forward(self, audio_inputs):
        audio_inputs=self.flatten(audio_inputs)
        #print(audio_inputs.shape)
        audio_inputs=self.rnn(audio_inputs.unsqueeze(0))
        #print(audio_inputs.shape)
        audio_inputs=self.linear1(audio_inputs[0])
        audio_inputs=self.linear2(audio_inputs)
        audio_inputs=self.linear3(audio_inputs)  
        #audio_inputs=self.linear5(audio_inputs)       
        #exp_0=exp_0.contiguous().view(exp_0.shape[0], 1, exp_0.shape[1])
        #audio_inputs=audio_inputs.contiguous().view(audio_inputs.shape[0], 1, audio_inputs.shape[1])
        #concat_z = torch.cat([audio_inputs, exp_0], dim=1)
        #concat_z=self.flatten(concat_z)
        #concat_z=self.linear4(concat_z)
        #out=self.linear5(concat_z)
        return audio_inputs
        
class Audio2Exp(nn.Module):
    def __init__(self, hidden_size=128):
        super(Audio2Exp, self).__init__()
        '''
        #self.conv_layer_exp = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        #self.pool_layer_exp = nn.MaxPool1d(kernel_size=3, stride=2)
        #self.linear_layer_exp = nn.Linear(48, 29)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=6, stride=1, padding=1)
        #self.conv_layer_exp = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=3)
    
        # the input map is 1 x 12 x 28       
        self.block1 = BasicBlock_AU(2, 8, (1,3), stride=(1,2),maxpool=(1,1)) # 3 x 12 x 13
        self.block2 = BasicBlock_AU(8, 32, kernel_size=(1,3), stride=(1,2),maxpool=(1,1)) # 8 x 12 x 6
        self.block3 = BasicBlock_AU(32, 64, kernel_size=(1,3), stride=(1,2),maxpool=(1,2)) # 16 x 12 x 1
        self.block4 = BasicBlock_AU(64, 128, kernel_size=(3,1), stride=(2,1),maxpool=(1,1)) # 32 x 5 x 1
        self.block5 = BasicBlock_AU(128, 256, kernel_size=(3,1), stride=(2,1),maxpool=(2,1)) # 32 x 1 x 1 
        self.rnn = RNNModel(256, hidden_size)
        self.fc1 = nn.Sequential(nn.Linear(256,128), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5)) # 128
        self.fc2 = nn.Sequential(nn.Linear(128,127), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5)) # 128
        '''
        self.flatten = nn.Flatten()
        self.rnn = RNNModel(16 * 29, 8 * 29)
        self.linear1 = nn.Sequential(nn.Linear(16 * 29, 8 * 29), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        self.linear2 = nn.Sequential(nn.Linear(8 * 29, 4*29), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        self.linear3 = nn.Sequential(nn.Linear(4 * 29, 79), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        #self.linear4 = nn.Sequential(nn.Linear(79*2, 79), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        self.linear5 = nn.Sequential(nn.Linear(79, 30), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
              
    def forward(self, audio_inputs):
        audio_inputs=self.flatten(audio_inputs)
        #print(audio_inputs.shape)
        audio_inputs=self.rnn(audio_inputs.unsqueeze(0))
        #print(audio_inputs.shape)
        audio_inputs=self.linear1(audio_inputs[0])
        audio_inputs=self.linear2(audio_inputs)
        audio_inputs=self.linear3(audio_inputs)  
        audio_inputs=self.linear5(audio_inputs)       
        #exp_0=exp_0.contiguous().view(exp_0.shape[0], 1, exp_0.shape[1])
        #audio_inputs=audio_inputs.contiguous().view(audio_inputs.shape[0], 1, audio_inputs.shape[1])
        #concat_z = torch.cat([audio_inputs, exp_0], dim=1)
        #concat_z=self.flatten(concat_z)
        #concat_z=self.linear4(concat_z)
        #out=self.linear5(concat_z)
        return None,audio_inputs
        '''
        #print(audio_inputs.shape)
        batchsize=audio_inputs.shape[0]
        exp_0=exp_0.contiguous().view(exp_0.shape[0], 1, 1, exp_0.shape[1])
        output_tensor = self.conv1(exp_0.squeeze(dim=2))
        output_tensor = self.conv2(output_tensor)
        output_tensor = self.conv3(output_tensor)
        output_tensor = output_tensor.unsqueeze(dim=1)
        #output_tensor = self.pool_layer_exp(output_tensor)
        #output_tensor = output_tensor.view(2, -1)
        #output_tensor = self.linear_layer_exp(output_tensor)
        #output_tensor = output_tensor.view(2, 1, 16, 29)
        
        #seq_len=audio_inputs.shape[1]
        audio_inputs = audio_inputs.contiguous().view(audio_inputs.shape[0], 1, audio_inputs.shape[1],audio_inputs.shape[2])
        #print(output_tensor.shape,audio_inputs.shape)
        concat_z = torch.cat([output_tensor , audio_inputs], dim=1)
        out = self.block1(concat_z)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        out = out.contiguous().view(out.shape[0], -1)
        out = out.contiguous().view(batchsize, -1)
        out = self.rnn(out)
        rnn_out = out.contiguous().view(batchsize, -1)
        out = self.fc1(rnn_out)
        out = self.fc2(out)

        return rnn_out,out
        '''




def load_ckpt(model, ckpt_path):
    old_state_dict = ckpt_path
    cur_state_dict = model.state_dict()
    for param in cur_state_dict:

        old_param = param
        if old_param in old_state_dict and cur_state_dict[param].size()==old_state_dict[old_param].size():
            print("loading param: ", param)
            model.state_dict()[param].data.copy_(old_state_dict[old_param].data)
        else:
            print("warning cannot load param: ", param)
class Trainer(object):
    def __init__(self,config,data_loader):
        '''
        Training instance of headnerf
        '''
        self.config = config ## load configuration
        #self.propress = Preprocesser("cuda")

        ####load configurations####################
        # data params 
        if config.is_train:
            self.train_loader = data_loader[0]
            self.val_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            print(f'Load {self.num_train} data samples')
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
            print(f'Load {self.num_test} data samples')
        self.batch_size = config.batch_size
        self.use_gt_camera = config.use_gt_camera
        self.include_eye_gaze = config.include_eye_gaze
        self.eye_gaze_dim = config.eye_gaze_dimension
        self.use_6D_rotation = config.gaze_D6_rotation
        if self.eye_gaze_dim%2 == 1:
            #we need eye_gaze_dim to be even number
            raise Exception("eye_gaze_dim expected to be even number!")
        if self.use_6D_rotation and self.eye_gaze_dim%6!=0:
            raise Exception("eye_gaze_dim expected to be 6n when using 6D rotation representation!")

        self.eye_gaze_scale_factor = config.eye_gaze_scale_factor
        self.disentangle = config.eye_gaze_disentangle

        # training params
        self.epochs = config.epochs  # the total epoch to train
        self.start_epoch = 0
        self.lr = config.init_lr
        self.lr_patience = config.lr_patience
        self.lr_decay_factor = config.lr_decay_factor
        self.resume = config.resume


        # misc params
        self.use_gpu = config.use_gpu
        self.gpu_id = config.gpu_id
        self.ckpt_dir = config.ckpt_dir  # output dir
        self.print_freq = config.print_freq
        self.train_iter = 0
        self.pre_trained_model_path = config.pre_trained_model_path
        self.headnerf_options = config.headnerf_options

        # configure tensorboard logging
        log_dir = './logs/' + os.path.basename(os.getcwd())
        if os.path.exists(log_dir) and os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        #self.Audio2Exp = Audio2Exp().cuda()




        #wav2lip

        self.Audio2style = Audio2style()        





        '''
        current_root_path = os.path.split(sys.argv[0])[0]
        sadtalker_paths = init_path('./checkpoints', os.path.join(current_root_path, 's_config'), 256, False, 'crop')
        self.sadtalkerexp = CropAndExtract(sadtalker_paths,'cpu')
        #sadtaker
        self.netG = SimpleWrapperV2()
        self.netG = self.netG.to(device)
        for param in self.netG.parameters():
            param.requires_grad = True
        self.netG.train()
        current_root_path = os.path.split(sys.argv[0])[0]
        sadtalker_paths = init_path('./checkpoints', os.path.join(current_root_path, 's_config'), 256, False, 'crop')
        try:
            if sadtalker_paths['use_safetensor']:
                checkpoints = safetensors.torch.load_file(sadtalker_paths['checkpoint'])
                self.netG.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2exp'))
            else:
                load_cpk(sadtalker_paths['audio2exp_checkpoint'], model=self.netG, device=device)
        except:
            raise Exception("Failed in loading audio2exp_checkpoint")
        '''
        #self.audio2exp_model = Audio2Exp_sadtaker(self.netG, cfg_exp, device=device, prepare_training_loss=False)
        #self.audio2exp_model = self.audio2exp_model.to(device)
        #for param in self.audio2exp_model.parameters():
        #    param.requires_grad = True
        #self.audio2exp_model.train()
        #grad_vars = list(self.audio2exp_model.parameters()) + list(netG.parameters())
        #self.optimizer_auds2exp=optim.Adam(params=list(self.netG.parameters()), lr=0.0000001, betas=(0.5, 0.999))
        self.optimizer_audio2style=optim.Adam(params=list(self.Audio2style.parameters()), lr=0.0000001, betas=(0.5, 0.999))
 
        self.device = device





        #liploss
        #cfg = parse_args()
        
        #self.liptrainer = lipTrainer()







        #build model
        if self.headnerf_options:
            
            check_dict = torch.load(self.headnerf_options, map_location=torch.device("cpu"))
            check_dict1 = torch.load('/home/logs/ckpt/epoch_7_fusemodel_ckpt.pth.tar', map_location=torch.device("cpu"))

            para_dict = check_dict["para"]
            self.opt = BaseOptions(para_dict)

            self.model = HeadNeRFNet(self.opt, include_vd=False, hier_sampling=False,include_gaze=self.include_eye_gaze,eye_gaze_dim=self.eye_gaze_dim)  
            #self.AudNet = AudioNet(64, 16).to(device)
            self._load_model_parameter(check_dict,check_dict1)
            print(f'load model parameter from {self.headnerf_options},set include_eye gaze to be {self.include_eye_gaze}')
        else:
            self.opt = BaseOptions()
            self.model = HeadNeRFNet(self.opt, include_vd=False, hier_sampling=False,include_gaze=self.include_eye_gaze,eye_gaze_dim=self.eye_gaze_dim)   
            #self.AudNet = AudioNet(64, 16).to(device)
            print(f'Train model from scratch, set include_eye gaze to be {self.include_eye_gaze}')     
        
        ##device setting
        if self.use_gpu and torch.cuda.device_count() > 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")  
            if self.gpu_id >=0 :
                torch.cuda.set_device(self.gpu_id)
            gpu_id = torch.cuda.current_device()
            self.device = torch.device("cuda:%d" % gpu_id)
            self.model.cuda()
            #self.AudNet.cuda()
            print(f'GPU {str(gpu_id).zfill(2)} name:{torch.cuda.get_device_name(gpu_id)}')
        else:
            self.device = torch.device("cpu")
            




        
        #initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr)
        #self.optimizer_Aud = torch.optim.Adam(params=list(self.AudNet.parameters()), lr=5e-4, betas=(0.9, 0.999))
        #self.optimizer_auds2exp=optim.Adam(self.netG.parameters(), lr=0.000001, betas=(0.5, 0.999))
        self.scheduler = StepLR(
            self.optimizer, step_size=self.lr_patience, gamma=self.lr_decay_factor)

        self._build_tool_funcs()

        if self.resume:
            self.load_checkpoint(self.resume)
            
    def _load_model_parameter(self,check_dict,check_dict1):
        #dealing with extended model when include eye gaze input
        if self.include_eye_gaze:
            #weight list contains keys that needs to be extended when include eye_gaze in headnerf
            weight_list = ["fg_CD_predictor.FeaExt_module_5.weight","fg_CD_predictor.FeaExt_module_0.weight"]
            #weight_list = ["fg_CD_predictor.FeaExt_module_5.weight", "fg_CD_predictor.RGB_layer_1.weight","fg_CD_predictor.FeaExt_module_0.weight"]
            for key in weight_list:
                r,c,_,_ = check_dict["net"][key].size()
                original_weight = check_dict["net"][key]
                extended_weight = torch.zeros((r,self.eye_gaze_dim,1,1))
                new_weight = torch.cat((original_weight,extended_weight),1)
                assert new_weight.size(1) == c + self.eye_gaze_dim
                check_dict["net"][key] = new_weight
            print(f'Eye gaze feature dimension: {self.eye_gaze_dim}')
        #self.model.load_state_dict(check_dict["net"])
        load_ckpt(self.model,check_dict["net"])
        #if 'auds2exp' not in check_dict:
        #self.netG.load_state_dict(check_dict1["auds2exp"])
        #else:
        #    self.Audio2Exp.load_state_dict(check_dict["auds2exp"])
        #if 'aud' in check_dict:
            #load_ckpt(self.AudNet,check_dict["aud"])
        #    self.AudNet.load_state_dict(check_dict["aud"])
        if 'audio2style' not in check_dict:
            self.Audio2style.load_state_dict(check_dict1["audio2style"])
        else:
            self.Audio2style.load_state_dict(check_dict["audio2style"])
        if 'optim_style' in check_dict:
            self.optimizer_audio2style.load_state_dict(check_dict["optim_style"])

    def _build_tool_funcs(self):
        self.loss_utils = HeadNeRFLossUtils(device=self.device)
        self.render_utils = RenderUtils(view_num=45, device=self.device, opt=self.opt)
        
        self.xy = self.render_utils.ray_xy.to(self.device).expand(self.batch_size,-1,-1)
        self.uv = self.render_utils.ray_uv.to(self.device).expand(self.batch_size,-1,-1)

    def build_code_and_cam_info_exp(self,data_info):
        #face_gaze = data_info['gaze'].float()
        mm3d_param = data_info['_3dmm']
        base_iden = mm3d_param['code_info']['base_iden'].squeeze(1)
        base_expr = mm3d_param['code_info']['base_expr'].squeeze(1)
        base_text = mm3d_param['code_info']['base_text'].squeeze(1)
        base_illu = mm3d_param['code_info']['base_illu'].squeeze(1)

        base_iden_i = mm3d_param['code_info_i']['base_iden'].squeeze(1)
        base_expr_i = mm3d_param['code_info_i']['base_expr'].squeeze(1)
        base_text_i = mm3d_param['code_info_i']['base_text'].squeeze(1)
        base_illu_i = mm3d_param['code_info_i']['base_illu'].squeeze(1)
        return base_expr,base_expr_i
        '''




        if self.include_eye_gaze:
            self.face_gaze = face_gaze.clone()
            if self.use_6D_rotation:
                ##the transformation is non-linear cannot be directly scaled
                face_gaze = data_info['gaze_6d'].float() * self.eye_gaze_scale_factor
            else:
                face_gaze = (face_gaze) * self.eye_gaze_scale_factor

            face_gaze = face_gaze.repeat(1,self.eye_gaze_dim//face_gaze.size(1))
            shape_code = torch.cat([base_iden, base_expr,face_gaze], dim=-1)
            appea_code = torch.cat([base_text, base_illu], dim=-1) ##test
        else:
            shape_code = torch.cat([base_iden, base_expr], dim=-1)
            appea_code = torch.cat([base_text, base_illu], dim=-1) ##test
            shape_code_i = torch.cat([base_iden_i, base_expr_i], dim=-1)
            appea_code_i = torch.cat([base_text_i, base_illu_i], dim=-1) ##test        

        if self.use_gt_camera:
            base_Rmats = data_info['camera_parameter']['cam_rotation'].clone().detach().float().to(self.device)
            base_Tvecs = data_info['camera_parameter']['cam_translation'].clone().detach().float().to(self.device)
            batch_inv_inmat = mm3d_param['cam_info']["batch_inv_inmats"].squeeze(1)
        else:
            base_Rmats = mm3d_param['cam_info']["batch_Rmats"].squeeze(1)
            base_Tvecs = mm3d_param['cam_info']["batch_Tvecs"].squeeze(1)
            batch_inv_inmat = mm3d_param['cam_info']["batch_inv_inmats"].squeeze(1)
        

        cam_info = {
                "batch_Rmats": base_Rmats.to(self.device),
                "batch_Tvecs": base_Tvecs.to(self.device),
                "batch_inv_inmats": batch_inv_inmat.to(self.device)
            }
        code_info = {
            "bg_code": None, 
            "shape_code":shape_code.to(self.device), 
            "appea_code":appea_code.to(self.device), 
        }
        code_info_i = {
            "bg_code": None, 
            "shape_code_i":shape_code_i.to(self.device), 
            "appea_code_i":appea_code_i.to(self.device), 
        }
        return code_info,code_info_i,cam_info
        '''


    def build_code_and_cam_info(self,data_info,auds_exp):
        #face_gaze = data_info['gaze'].float()
        mm3d_param = data_info['_3dmm']
        base_iden = mm3d_param['code_info']['base_iden'].squeeze(1)
        base_expr = mm3d_param['code_info']['base_expr'].squeeze(1)#auds_exp
        base_text = mm3d_param['code_info']['base_text'].squeeze(1)
        base_illu = mm3d_param['code_info']['base_illu'].squeeze(1)

        base_iden_i = mm3d_param['code_info_i']['base_iden'].squeeze(1)
        base_expr_i = mm3d_param['code_info_i']['base_expr'].squeeze(1)
        base_text_i = mm3d_param['code_info_i']['base_text'].squeeze(1)
        base_illu_i = mm3d_param['code_info_i']['base_illu'].squeeze(1)

        
        #bian=[0,1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,59,61,63,64,66,68,69,73,77]
        #print(data_info['sad_exp'].shape)
        # print(base_expr.shape, auds_exp.shape)
        #for ii in range(len(bian)):
        base_expr[:,:64]=auds_exp


        if self.include_eye_gaze:
            self.face_gaze = face_gaze.clone()
            if self.use_6D_rotation:
                ##the transformation is non-linear cannot be directly scaled
                face_gaze = data_info['gaze_6d'].float() * self.eye_gaze_scale_factor
            else:
                face_gaze = (face_gaze) * self.eye_gaze_scale_factor

            face_gaze = face_gaze.repeat(1,self.eye_gaze_dim//face_gaze.size(1))
            shape_code = torch.cat([base_iden, base_expr,face_gaze], dim=-1)
            appea_code = torch.cat([base_text, base_illu], dim=-1) ##test
        else:
            shape_code = torch.cat([base_iden.cuda(), base_expr.cuda()], dim=-1)
            appea_code = torch.cat([base_text, base_illu], dim=-1) ##test
            shape_code_i = torch.cat([base_iden_i, base_expr_i], dim=-1)
            appea_code_i = torch.cat([base_text_i, base_illu_i], dim=-1) ##test        

        if self.use_gt_camera:
            base_Rmats = data_info['camera_parameter']['cam_rotation'].clone().detach().float().to(self.device)
            base_Tvecs = data_info['camera_parameter']['cam_translation'].clone().detach().float().to(self.device)
            batch_inv_inmat = mm3d_param['cam_info']["batch_inv_inmats"].squeeze(1)
        else:
            base_Rmats = mm3d_param['cam_info']["batch_Rmats"].squeeze(1)
            base_Tvecs = mm3d_param['cam_info']["batch_Tvecs"].squeeze(1)
            batch_inv_inmat = mm3d_param['cam_info']["batch_inv_inmats"].squeeze(1)
        

        cam_info = {
                "batch_Rmats": base_Rmats.to(self.device),
                "batch_Tvecs": base_Tvecs.to(self.device),
                "batch_inv_inmats": batch_inv_inmat.to(self.device)
            }
        code_info = {
            "bg_code": None, 
            "shape_code":shape_code.to(self.device), 
            "appea_code":appea_code.to(self.device), 
        }
        code_info_i = {
            "bg_code": None, 
            "shape_code_i":shape_code_i.to(self.device), 
            "appea_code_i":appea_code_i.to(self.device), 
        }
        return code_info,code_info_i,cam_info
    
    def train(self):
        self.logging_config('./logs')
        for epoch in range(self.start_epoch,self.epochs):
            print(
                '\nEpoch: {}/{} - base LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.lr)
            )
            self.cur_epoch = epoch
            #Training
            self.model.train()
            self.train_one_epoch(epoch,self.train_loader)
            #print("hhh")

            add_file_name = 'epoch_' + str(epoch)

            para_dict={}
            para_dict["featmap_size"] = self.opt.featmap_size
            para_dict["featmap_nc"] = self.opt.featmap_nc 
            para_dict["pred_img_size"] = self.opt.pred_img_size

            val_dic = self.validation(epoch)

            add_file_name+= "_%.2f_%.2f_%.2f" % (val_dic['SSIM'],val_dic['PSNR'],val_dic['LPIPS'])
            
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 #'auds2exp': self.netG.state_dict(),
                 #'optim_exp': self.optimizer_auds2exp.state_dict(),
                 'optim_style': self.optimizer_audio2style.state_dict(),
                 'audio2style':self.Audio2style.state_dict(),
                 'net': self.model.state_dict(),
                 #'aud': self.AudNet.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 #'optim_state_aud': self.optimizer_Aud.state_dict(),
                 'scheule_state': self.scheduler.state_dict(),
                 'para':para_dict
                 }, add=add_file_name
            )

            val_dic['ckpt_name'] = add_file_name
            self.logging_config('./logs',val_dic)

            self.scheduler.step() 


        self.writer.close()
    
    def eye_gaze_displacement(self,data_info,code_info,cam_info):
        if self.use_6D_rotation:
            face_gaze_new = data_info['gaze_disp_d6'].float() * self.eye_gaze_scale_factor
        else:
            face_gaze_new = data_info['gaze_disp'].float() * self.eye_gaze_scale_factor
            
        code_info['shape_code'][:,-self.eye_gaze_dim:] = face_gaze_new.repeat(1,self.eye_gaze_dim//face_gaze_new.size(1))
        
        pred_dict_p = self.model( "train", self.xy, self.uv,  **code_info, **cam_info)

        return pred_dict_p,face_gaze_new


    def train_one_epoch(self, epoch, data_loader, is_train=True):
        loop_bar = tqdm(enumerate(data_loader), leave=False, total=len(data_loader))
        l_fen = np.zeros(79)
        count=0
        
        for iter,data_info in loop_bar:

            with torch.set_grad_enabled(True):
                
                #code_info,code_info_i,cam_info = self.build_code_and_cam_info(data_info)
                #base_expr,base_expr_i=self.build_code_and_cam_info_exp(data_info)
                #bian=[0,1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,59,61,63,64,66,68,69,73,77]
                #print(len(bian))
                #base_expr_ = torch.zeros((base_expr.shape[0],len(bian)))
                #base_expr_i_ = torch.zeros((base_expr_i.shape[0],len(bian)))
                #for jjj in range(len(bian)):
                #   base_expr_[:,jjj]=base_expr[:,bian[jjj]]
                #   base_expr_i_[:,jjj]=base_expr_i[:,bian[jjj]]
                #base_expr[:,:64]=data_info['sad_exp'].squeeze(1)




                '''
                img_batch, mel_batch, frames, coords=data_info['wav_gen1'],data_info['wav_gen2'],data_info['wav_gen3'],data_info['wav_gen4']
                auds_exp=[]
                print=(torch.stack(img_batch).shape,torch.stack(frames).shape)
                for j in range(self.batch_size):
                    img_batch_ = torch.FloatTensor(np.transpose(img_batch[j].float(), (0, 3, 1, 2))).to(device)
                    mel_batch_ = torch.FloatTensor(np.transpose(mel_batch[j].float(), (0, 3, 1, 2))).to(device)
                    pred = self.wav_model(mel_batch_, img_batch_)
                    pred = pred.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255.
                    for p, f, c in zip(pred, frames[j], coords.squeeze(1)[j]):
                        y1, y2, x1, x2 = c
                        p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                        f[y1:y2, x1:x2] = p
                        cv2.imwrite(os.path.join('/home/temp/' +str(j)+ '.png'),f)
                    auds_exp.append(torch.from_numpy(self.sadtalkerexp.generate('/home/temp/' +str(j)+ '.png', '/home/dataset/Obama/sadtalker/', 'crop', source_image_flag=True, pic_size=256)))
                auds_exp=torch.stack(auds_exp)
                '''
                

                mel_batch=data_info['mel_batch']
                #print(mel_batch.shape) 2 1 1 80 16
                audiostyle=self.Audio2style(mel_batch.squeeze(1).squeeze(1))
                auds_exp=data_info['wav_gen'].squeeze(1)
                #print(auds_exp.shape)
                code_info,code_info_i,cam_info = self.build_code_and_cam_info(data_info,auds_exp.cuda())
                pred_dict = self.model( "train", self.xy, self.uv,audiostyle.cuda(),  **code_info, **cam_info)
                
                coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]  
                #coarse_fg_rgb__ = coarse_fg_rgb[:,:,256:512,:]
                
                # print(coarse_fg_rgb.shape) # 2 3 512 512
                '''
                for filename in os.listdir('/fs1/home/tjuvis_2022/linshi/'):
                    file_path = os.path.join('/fs1/home/tjuvis_2022/linshi/', filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                for j in range(self.batch_size):
                    coarse_fg_rgb1 = (coarse_fg_rgb[j].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
                    res_img=cv2.cvtColor(coarse_fg_rgb1, cv2.COLOR_BGR2RGB)
                    cv2.imwrite('/fs1/home/tjuvis_2022/linshi/' + str(j) + '.jpg',res_img)  
                extract_ds_cmd = 'python DataProcess/Gen_Landmark.py --img_dir "/fs1/home/tjuvis_2022/linshi" '
                os.system(extract_ds_cmd)             
    
                extract_ds_cmd = 'python Fitting3DMM/FittingNL3DMM.py --img_size 512 --intermediate_size 256 --batch_size 2 --img_dir "/fs1/home/tjuvis_2022/linshi" '
                os.system(extract_ds_cmd)  
                j=0    
                base_expr_gen=[]
                for j in range(self.batch_size):    
                    dmm_name=str(j) + '_nl3dmm.pkl'
                    para_3dmm_path = os.path.join('/fs1/home/tjuvis_2022/linshi',dmm_name)
                    with open(para_3dmm_path, "rb") as f: nl3dmm_para_dict = pkl.load(f)
                    base_code = nl3dmm_para_dict["code"].float().detach().unsqueeze(0).cuda()
                    base_expr_gen.append(base_code[:, 100:179].squeeze(1))             
                base_expr_gen=torch.stack(base_expr_gen) 
                '''         
                if self.disentangle:
                    disp_pred_dict,disp_gaze = self.eye_gaze_displacement(data_info,code_info,cam_info)
                else:
                    disp_pred_dict = None

                gt_img = data_info['img'].squeeze(1); mask_img = data_info['img_mask'].squeeze(1);eye_mask=None#data_info['eye_mask'].squeeze(1)

                '''
                for p in range(data_info['idx'].shape[0]):
                    coarse_fg_rgb_ = pred_dict["coarse_dict"]["merge_img"]
                    coarse_fg_rgb_ = (coarse_fg_rgb_[p].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
                    res_img=cv2.cvtColor(coarse_fg_rgb_, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join('/home/test/Macron/gen/',str(data_info['idx'][p].item()) + '.png'),res_img)  
                    #coarse_fg_rgb = coarse_fg_rgb[:, :, [2, 1, 0]]
                    gt_img_ = (gt_img[p].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
                    res_img=cv2.cvtColor(gt_img_, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join('/home/test/Macron/gt/',str(data_info['idx'][p].item()) + '.png'),res_img)                   '''
                
                
                ##compute head loss
                batch_loss_dict = self.loss_utils.calc_total_loss(
                    delta_cam_info=None, opt_code_dict=None, pred_dict=pred_dict, disp_pred_dict=disp_pred_dict,
                    gt_rgb=gt_img.to(self.device), mask_tensor=mask_img.to(self.device),eye_mask_tensor=eye_mask#.to(self.device)
                )
            total_loss=(batch_loss_dict["total_loss"])
            self.optimizer_audio2style.zero_grad()
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.optimizer_audio2style.step()

            

            if isnan(batch_loss_dict["head_loss"].item()):
                import warnings
                warnings.warn('nan found in batch loss !! please check output of HeadNeRF')
            if self.disentangle:
                loop_bar.set_description("Opt, Head_loss/Img_disp/Lm_disp: %.6f / %.6f / %.6f" % (batch_loss_dict["head_loss"].item(),batch_loss_dict["image_disp_loss"].item(),batch_loss_dict["lm_disp_loss"].item()) )  
            else:
                loop_bar.set_description("Opt, Head_loss: %.6f" % (batch_loss_dict["head_loss"].item()) )  
                #loop_bar.set_description("Opt, Head_loss: %.6f, exp_loss: %.6f, lm_loss: %.6f, lip_loss: %.6f" % (batch_loss_dict["head_loss"].item(), exp_loss.item(), lm_loss.item(), 10000*liploss.item()))
                #loop_bar.set_description("Opt, Head_loss: %.6f" % (batch_loss_dict["head_loss"].item()))  
            
        #j=0
        #for j in range(79):
        #    print(j," : ",l_fen[j]/count)


                
    def validation(self,epoch):
        self.model.eval()
        output_dict = {
        'SSIM':0,
        'PSNR':0,
        'LPIPS':0
        }
        count = 0
        loop_bar = enumerate(self.val_loader)
        xy = self.render_utils.ray_xy.to(self.device).expand(self.val_loader.batch_size,-1,-1)
        uv = self.render_utils.ray_uv.to(self.device).expand(self.val_loader.batch_size,-1,-1)
        #self.netG.eval()
        for iter,data_info in loop_bar:
            with torch.set_grad_enabled(False):
                '''
                base_expr,base_expr_i=self.build_code_and_cam_info_exp(data_info)
                #code_info,code_info_i,cam_info = self.build_code_and_cam_info(data_info)
                bian=[0,1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,59,61,63,64,66,68,69,73,77]
                #print(len(bian))
                base_expr_ = torch.zeros((base_expr.shape[0],len(bian)))
                base_expr_i_ = torch.zeros((base_expr_i.shape[0],len(bian)))
                for jjj in range(len(bian)):
                   base_expr_[:,jjj]=base_expr[:,bian[jjj]]
                   base_expr_i_[:,jjj]=base_expr_i[:,bian[jjj]]
                '''
                
                mel_batch=data_info['mel_batch']
                #print(mel_batch.shape) 2 1 1 80 16
                audiostyle=self.Audio2style(mel_batch.squeeze(1).squeeze(1))                
                auds_exp=data_info['wav_gen'].squeeze(1)
                code_info,code_info_i,cam_info = self.build_code_and_cam_info(data_info,auds_exp.cuda())
                #auds_val = self.AudNet(auds.cuda()).unsqueeze(0)
                pred_dict = self.model( "test", xy, uv,audiostyle.cuda(),  **code_info, **cam_info)

                
                gt_img = data_info['img'].squeeze(1); mask_img = data_info['img_mask'].squeeze(1)#;eye_mask=data_info['eye_mask'].squeeze(1)

                eval_metrics = calc_eval_metrics(pred_dict=pred_dict,gt_rgb=gt_img.to(self.device),mask_tensor=mask_img.to(self.device),vis=False)
                
                output_dict['SSIM'] += eval_metrics['SSIM']
                output_dict['PSNR'] += eval_metrics['PSNR']
                output_dict['LPIPS'] += eval_metrics['LPIPS']
                count+=1

            #print(self.print_freq,iter)
            #if iter % self.print_freq == 0 and iter != 0:
            '''
            for p in range(data_info['idx'].shape[0]):
                coarse_fg_rgb_ = pred_dict["coarse_dict"]["merge_img"]
                coarse_fg_rgb_ = (coarse_fg_rgb_[p].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
                res_img=cv2.cvtColor(coarse_fg_rgb_, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join('/home/test/Macron/gen/',str(data_info['idx'][p].item()) + '.png'),res_img)  
                #coarse_fg_rgb = coarse_fg_rgb[:, :, [2, 1, 0]]
                gt_img_ = (gt_img[p].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
                res_img=cv2.cvtColor(gt_img_, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join('/home/test/Macron/gt/',str(data_info['idx'][p].item()) + '.png'),res_img) 
            '''
            self._display_current_rendered_image(pred_dict,gt_img,iter)
        #self.netG.train()
        output_dict['SSIM'] /= count
        output_dict['PSNR'] /= count
        output_dict['LPIPS'] /= count
        print("Evaluation Metrics: SSIM: %.4f  PSNR: %.4f  LPIPS: %.4f" % (output_dict['SSIM'],output_dict['PSNR'],output_dict['LPIPS']))
        return output_dict



    def save_checkpoint(self, state, add=None):
        """
        Save a copy of the model
        """
        if add is not None:
            filename = add + '_ckpt.pth.tar'
        else:
            filename ='ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        print('save file to: ', filename)

    def load_checkpoint(self, input_file_path='./ckpt/ckpt.pth.tar', is_strict=True):
        """
        Load the copy of a model.
        """
        print('load the pre-trained model: ', input_file_path)
        ckpt = torch.load(input_file_path)
        
        ckpt1 = torch.load('/home/logs/ckpt/epoch_25_fusemodel_ckpt.pth.tar')
        #ckpt1 = torch.load('/fs1/home/tjuvis_2022/pt/epoch_25_ckpt.pth.tar')

        # load variables from checkpoint
        self.model.load_state_dict(ckpt['net'], strict=is_strict)
        #self.AudNet.load_state_dict(ckpt['aud'], strict=is_strict)
        #if 'auds2exp' in ckpt:
        #    self.Audio2Exp.load_state_dict(ckpt['auds2exp'], strict=is_strict)
        #    self.optimizer_auds2exp.load_state_dict(ckpt['optim_exp'])
        #else:
        #self.netG.load_state_dict(ckpt1['auds2exp'], strict=is_strict)
        if 'audio2style' not in check_dict:
            self.Audio2style.load_state_dict(check_dict1["audio2style"])   
        else:
            self.Audio2style.load_state_dict(check_dict["audio2style"])
        self.optimizer_auds2exp.load_state_dict1(ckpt['optim_exp'])
        #self.optimizer_audio2style.load_state_dict1(ckpt['optim_style'])
        self.optimizer.load_state_dict(ckpt['optim_state'])
        
            
        #self.optimizer_Aud.load_state_dict(ckpt['optim_state_aud'])
        self.scheduler.load_state_dict(ckpt['scheule_state'])
        self.start_epoch = ckpt['epoch'] 

        print(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                input_file_path, ckpt['epoch'])
        )

    def _display_current_rendered_image(self,pred_dict,img_tensor,iter):
        coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
        coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        #coarse_fg_rgb = coarse_fg_rgb[:, :, [2, 1, 0]]
        gt_img = (img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        res_img = np.concatenate([gt_img, coarse_fg_rgb], axis=1)

        
        log_path = './logs/temp_image/' + 'epoch' + str(self.cur_epoch)
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)
        res_img=cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(log_path,str(iter).zfill(6) + 'iter_image.png'),res_img)
        print(f'Save temporary rendered image to {log_path}')

        # cv2.imshow('current rendering', res_img)
        # cv2.waitKey(0) 
        # #closing all open windows 
        # cv2.destroyAllWindows() 
    
    def logging_config(self,log_path,val_dict={}):
        from datetime import datetime
        if not val_dict :
            now = datetime.now()       
            print("now =", now)
            self.logger = log(path=log_path,file=f'{now}_training_log_file.logs')

            config_list=['batch_size','init_lr','epochs','ckpt_dirs','include_eye_gaze','eye_gaze_dimension','gaze_D6_rotation','eye_gaze_scale_factor','comment']
            self.logger.info("----Training configuration----")
            for k,v in self.config.__dict__.items():
                if k in config_list:
                    self.logger.info(str(k) + ' : ' + str(v))
            self.logger.info("--------------------------------------------------")
        else: 
            self.logger.info("Evaluation Results")
            for k,v in val_dict.items():
                self.logger.info(str(k) + ' = ' + str(v))
            self.logger.info("--------------------------------------------------")


        


if __name__ == '__main__':
    check_dict = torch.load("/home/TrainedModels/model_Reso32.pth", map_location=torch.device("cpu"))
    para_dict = check_dict["para"]
    opt = BaseOptions(para_dict)
    model = HeadNeRFNet(opt, include_vd=False, hier_sampling=False,include_gaze=True)  
    import ipdb
    ipdb.set_trace()
    model.load_state_dict(check_dict["net"])


