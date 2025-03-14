from os.path import join
import os
import torch
import numpy as np
from NetWorks.HeadNeRFNet import HeadNeRFNet
import cv2
from HeadNeRFOptions import BaseOptions
from Utils.HeadNeRFLossUtils import HeadNeRFLossUtils
from Utils.RenderUtils import RenderUtils
import pickle as pkl
import time
from glob import glob
from tqdm import tqdm
import imageio
import random
import argparse
from tool_funcs import put_text_alignmentcenter
import torch.nn as nn
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
# sadtalker
import numpy as np
import cv2, os, sys, torch
from tqdm import tqdm
from PIL import Image 
import os

from tqdm import tqdm
import torch
import numpy as np
import random
import scipy.io as scio
import s_utils.audio as audio
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
def get_data(audio_path, device, ref_eyeblink_coeff_path, still=False, idlemode=False, length_of_audio=False, use_blink=True):

    syncnet_mel_step_size = 16
    fps = 25

    #pic_name = os.path.splitext(os.path.split(first_coeff_path)[-1])[0]
    audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]

    
    if idlemode:
        num_frames = int(length_of_audio * 25)
        indiv_mels = np.zeros((num_frames, 80, 16))
    else:
        wav = audio.load_wav(audio_path, 16000) 
        wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
        wav = crop_pad_audio(wav, wav_length)
        orig_mel = audio.melspectrogram(wav).T
        spec = orig_mel.copy()         # nframes 80
        indiv_mels = []
        
        
        #print(num_frames)

        for i in tqdm(range(num_frames), 'mel:'):
            start_frame_num = i-2
            start_idx = int(80. * (start_frame_num / float(fps)))
            end_idx = start_idx + syncnet_mel_step_size
            seq = list(range(start_idx, end_idx))
            seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
            m = spec[seq, :]
            indiv_mels.append(m.T)
        indiv_mels = np.asarray(indiv_mels)         # T 80 16

    ratio = generate_blink_seq_randomly(num_frames)      # T
    #source_semantics_path = first_coeff_path
    #source_semantics_dict = scio.loadmat(source_semantics_path)
    # ref_coeff = source_semantics_dict['coeff_3dmm'][:1,:70]         #1 70
    # ref_coeff = np.repeat(ref_coeff, num_frames, axis=0)

    if ref_eyeblink_coeff_path is not None:
        ratio[:num_frames] = 0
        refeyeblink_coeff_dict = scio.loadmat(ref_eyeblink_coeff_path)
        refeyeblink_coeff = refeyeblink_coeff_dict['coeff_3dmm'][:,:64]
        refeyeblink_num_frames = refeyeblink_coeff.shape[0]
        if refeyeblink_num_frames<num_frames:
            div = num_frames//refeyeblink_num_frames
            re = num_frames%refeyeblink_num_frames
            refeyeblink_coeff_list = [refeyeblink_coeff for i in range(div)]
            refeyeblink_coeff_list.append(refeyeblink_coeff[:re, :64])
            refeyeblink_coeff = np.concatenate(refeyeblink_coeff_list, axis=0)
            print(refeyeblink_coeff.shape[0])

        # ref_coeff[:, :64] = refeyeblink_coeff[:num_frames, :64] 
    
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0) # bs T 1 80 16

    if use_blink:
        ratio = torch.FloatTensor(ratio).unsqueeze(0)                       # bs T
    else:
        ratio = torch.FloatTensor(ratio).unsqueeze(0).fill_(0.) 
                               # bs T
    # ref_coeff = torch.FloatTensor(ref_coeff).unsqueeze(0)                # bs 1 70

    indiv_mels = indiv_mels.to(device)
    ratio = ratio.to(device)
    # ref_coeff = ref_coeff.to(device)
    #print(indiv_mels.shape)

    return indiv_mels, num_frames, ratio, audio_name
#wav2lip
from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, wav_audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch,face_detection
from wav_models import Wav2Lip
import platform
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
wav_checkpoint_path='/home/lipwav/pt/checkpoint_step000276000_.pth'
#wav_checkpoint_path='/home/lipwav/pt/checkpoint_step000348000.pth'
wav_face='/home/dataset/Obama/png/0.jpg'
wav_audio='/home/dataset/Obama/png/aud.wav'
wav_outfile='results/result_voice.mp4'
wav_static=False
wav_fps=25.
wav_pads=[0, 10, 0, 0]
wav_face_det_batch_size=16
wav_wav2lip_batch_size=1
wav_resize_factor=1
wav_crop=[0, -1, 0, -1]
wav_box=[-1, -1, -1, -1]
wav_rotate=False
wav_nosmooth=False
mel_step_size = 16
device =  'cpu'
wav_img_size = 96
if os.path.isfile(wav_face) and wav_face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    wav_static = True
def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    #print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()
def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device='cpu')

    batch_size = wav_face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = wav_pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not wav_nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 
def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if wav_box[0] == -1:
        if not wav_static:
            face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = wav_box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if wav_static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (wav_img_size, wav_img_size))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav_wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, wav_img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, wav_img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

def main(idx):
    if not os.path.isfile(wav_face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif wav_face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(wav_face)]
        fps = wav_fps

    else:
        video_stream = cv2.VideoCapture(wav_face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        #print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if wav_resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//wav_resize_factor, frame.shape[0]//wav_resize_factor))

            if wav_rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = wav_crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    wav = audio.load_wav(wav_audio, 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = idx
    start_idx = int(i * mel_idx_multiplier)
    if start_idx + mel_step_size > len(mel[0]):
        mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
        #break
    else:
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
    #mel_chunks=mel_chunks[2].unsqueeze(0)

    #print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]
    #print(len(mel_chunks))

    batch_size = wav_wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)
    return mel_chunks,batch_size,gen


        
class FittingImage(object):
    
    def __init__(self, model_path, save_root, gpu_id) -> None:
        super().__init__()
        self.model_path = model_path

        self.device = torch.device("cuda:%d" % gpu_id)
        self.save_root = save_root
        self.opt_cam = True
        self.view_num = 45
        self.duration = 3.0 / self.view_num
        self.model_name = os.path.basename(model_path)[:-4]

        self.build_info()
        self.build_tool_funcs()
        current_root_path = os.path.split(sys.argv[0])[0]
        sadtalker_paths = init_path('./checkpoints', os.path.join(current_root_path, 's_config'), 256, False, 'crop')
        self.sadtalkerexp = CropAndExtract(sadtalker_paths,'cpu')
        #self.batch1,self.batch2,self.batch3,self.batch4 = get_data(os.path.join(self.path, 'aud.wav'), "cpu", ref_eyeblink_coeff_path=None)

    def build_info(self):
        check_dict = torch.load(self.model_path, map_location=torch.device("cpu"))

        para_dict = check_dict["para"]
        self.opt = BaseOptions(para_dict) #just use the same feature size as the para_dict

        self.featmap_size = self.opt.featmap_size
        self.pred_img_size = self.opt.pred_img_size
        
        if not os.path.exists(self.save_root): os.mkdir(self.save_root)

        net = HeadNeRFNet(self.opt, include_vd=False, hier_sampling=False)        
        net.load_state_dict(check_dict["net"])
        self.Audio2style = Audio2style() 
        self.Audio2style.load_state_dict(check_dict["audio2style"])
        self.net = net.to(self.device)
        self.net.eval()


    def build_tool_funcs(self):
        self.loss_utils = HeadNeRFLossUtils(device=self.device)
        self.render_utils = RenderUtils(view_num=45, device=self.device, opt=self.opt)
        
        self.xy = self.render_utils.ray_xy
        self.uv = self.render_utils.ray_uv
    

    def load_data(self, img_path, mask_path, para_3dmm_path):
        
        #process imgs
        img_size = (self.pred_img_size, self.pred_img_size)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.0
        
        gt_img_size = img.shape[0]
        if gt_img_size != self.pred_img_size:
            img = cv2.resize(img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        if mask_img.shape[0] != self.pred_img_size:
            mask_img = cv2.resize(mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        img[mask_img < 0.5] = 1.0
        
        self.img_tensor = (torch.from_numpy(img).permute(2, 0, 1)).unsqueeze(0).to(self.device)
        self.mask_tensor = torch.from_numpy(mask_img[None, :, :]).unsqueeze(0).to(self.device)
        

       # load init codes from the results generated by solving 3DMM rendering opt.
        with open(para_3dmm_path, "rb") as f: nl3dmm_para_dict = pkl.load(f)
        base_code = nl3dmm_para_dict["code"].detach().unsqueeze(0).to(self.device)
        
        #ablation on 3DMM model codes
        IGNORE_3DMM_CODE=False
        if IGNORE_3DMM_CODE:
            base_code_zero = torch.zeros_like(base_code)
            base_code = base_code_zero
        
        self.base_iden = base_code[:, :self.opt.iden_code_dims]
        self.base_expr = base_code[:, self.opt.iden_code_dims:self.opt.iden_code_dims + self.opt.expr_code_dims]
        self.base_text = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims:self.opt.iden_code_dims 
                                                            + self.opt.expr_code_dims + self.opt.text_code_dims]
        self.base_illu = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims + self.opt.text_code_dims:]
        
        self.base_c2w_Rmat = nl3dmm_para_dict["c2w_Rmat"].detach().unsqueeze(0)
        self.base_c2w_Tvec = nl3dmm_para_dict["c2w_Tvec"].detach().unsqueeze(0).unsqueeze(-1)
        self.base_w2c_Rmat = nl3dmm_para_dict["w2c_Rmat"].detach().unsqueeze(0)
        self.base_w2c_Tvec = nl3dmm_para_dict["w2c_Tvec"].detach().unsqueeze(0).unsqueeze(-1)

        temp_inmat = nl3dmm_para_dict["inmat"].detach().unsqueeze(0)
        temp_inmat[:, :2, :] *= (self.featmap_size / gt_img_size)
        
        temp_inv_inmat = torch.zeros_like(temp_inmat)
        temp_inv_inmat[:, 0, 0] = 1.0 / temp_inmat[:, 0, 0]
        temp_inv_inmat[:, 1, 1] = 1.0 / temp_inmat[:, 1, 1]
        temp_inv_inmat[:, 0, 2] = -(temp_inmat[:, 0, 2] / temp_inmat[:, 0, 0])
        temp_inv_inmat[:, 1, 2] = -(temp_inmat[:, 1, 2] / temp_inmat[:, 1, 1])
        temp_inv_inmat[:, 2, 2] = 1.0
        
        self.temp_inmat = temp_inmat
        self.temp_inv_inmat = temp_inv_inmat

        if IGNORE_3DMM_CODE:
            batch_size = self.base_c2w_Rmat.size(0)
            self.base_c2w_Tvec = torch.zeros_like(self.base_c2w_Tvec)
            self.base_c2w_Rmat = torch.eye(3).repeat(batch_size,1).view(batch_size,3,3)

        self.cam_info = {
            "batch_Rmats": self.base_c2w_Rmat.to(self.device),
            "batch_Tvecs": self.base_c2w_Tvec.to(self.device),
            "batch_inv_inmats": self.temp_inv_inmat.to(self.device)
        }
        

    @staticmethod
    def eulurangle2Rmat(angles):
        batch_size = angles.size(0)
        
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXs = torch.eye(3, device=angles.device).view(1, 3, 3).repeat(batch_size, 1, 1)
        rotYs = rotXs.clone()
        rotZs = rotXs.clone()
        
        rotXs[:, 1, 1] = cosx
        rotXs[:, 1, 2] = -sinx
        rotXs[:, 2, 1] = sinx
        rotXs[:, 2, 2] = cosx
        
        rotYs[:, 0, 0] = cosy
        rotYs[:, 0, 2] = siny
        rotYs[:, 2, 0] = -siny
        rotYs[:, 2, 2] = cosy

        rotZs[:, 0, 0] = cosz
        rotZs[:, 0, 1] = -sinz
        rotZs[:, 1, 0] = sinz
        rotZs[:, 1, 1] = cosz
        
        res = rotZs.bmm(rotYs.bmm(rotXs))
        return res
    
    
    def build_code_and_cam(self,wav_sadtalkerexp):
        
        # code
        self.base_expr[:,:64]=wav_sadtalkerexp
        shape_code = torch.cat([self.base_iden + self.iden_offset, self.base_expr + self.expr_offset], dim=-1)
        appea_code = torch.cat([self.base_text, self.base_illu], dim=-1) + self.appea_offset
        
        opt_code_dict = {
            "bg":None,
            "iden":self.iden_offset,
            "expr":wav_sadtalkerexp,
            "appea":self.appea_offset
        }
        
        code_info = {
            "bg_code": None, 
            "shape_code":shape_code, 
            "appea_code":appea_code, 
        }

        
        #cam
        if self.opt_cam:
            delta_cam_info = {
                "delta_eulur": self.delta_EulurAngles, 
                "delta_tvec": self.delta_Tvecs
            }

            batch_delta_Rmats = self.eulurangle2Rmat(self.delta_EulurAngles)
            base_Rmats = self.cam_info["batch_Rmats"]
            base_Tvecs = self.cam_info["batch_Tvecs"]
            
            cur_Rmats = batch_delta_Rmats.bmm(base_Rmats)
            cur_Tvecs = batch_delta_Rmats.bmm(base_Tvecs) + self.delta_Tvecs
            
            batch_inv_inmat = self.cam_info["batch_inv_inmats"] #[N, 3, 3]    
            batch_cam_info = {
                "batch_Rmats": cur_Rmats,
                "batch_Tvecs": cur_Tvecs,
                "batch_inv_inmats": batch_inv_inmat
            }
            
        else:
            delta_cam_info = None
            batch_cam_info = self.cam_info


        return code_info, opt_code_dict, batch_cam_info, delta_cam_info
    
    
    @staticmethod
    def enable_gradient(tensor_list):
        for ele in tensor_list:
            ele.requires_grad = True


    def perform_fitting(self):
        self.delta_EulurAngles = torch.zeros((1, 3), dtype=torch.float32).to(self.device)
        self.delta_Tvecs = torch.zeros((1, 3, 1), dtype=torch.float32).to(self.device)

        self.iden_offset = torch.zeros((1, 100), dtype=torch.float32).to(self.device)
        self.expr_offset = torch.zeros((1, 79), dtype=torch.float32).to(self.device)
        self.appea_offset = torch.zeros((1, 127), dtype=torch.float32).to(self.device)

        if self.opt_cam:
            self.enable_gradient(
                [self.iden_offset, self.expr_offset, self.appea_offset, self.delta_EulurAngles, self.delta_Tvecs]
            )
        else:
            self.enable_gradient(
                [self.iden_offset, self.expr_offset, self.appea_offset]
            )
        
        init_learn_rate = 0.01
        
        step_decay = 300
        iter_num = 300
        
        params_group = [
            {'params': [self.iden_offset], 'lr': init_learn_rate * 1.5},
            {'params': [self.expr_offset], 'lr': init_learn_rate * 1.5},
            {'params': [self.appea_offset], 'lr': init_learn_rate * 1.0},
        ]
        
        if self.opt_cam:
            params_group += [
                {'params': [self.delta_EulurAngles], 'lr': init_learn_rate * 0.1},
                {'params': [self.delta_Tvecs], 'lr': init_learn_rate * 0.1},
            ]
            
        optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))
        lr_func = lambda epoch: 0.1 ** (epoch / step_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func) #adaptive learning rate
        
        gt_img = (self.img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        
        
        loop_bar = tqdm(range(iter_num), leave=True)
        for iter_ in loop_bar:
            with torch.set_grad_enabled(True):
                
                wav_sadtalkerexp=self.sadtalkerexp.generate('/home/dataset/Obama/lipwav/' + '0.jpg', '/home/dataset/Obama/sadtalker/', 'crop', source_image_flag=True, pic_size=256)
                mel_chunks,batch_size,gen=main(0)                
                #print(mel_chunks.shape)
                for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                                        total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
                    if i == 0:
                        model = load_model(wav_checkpoint_path)
                        #print ("Model loaded")
            
            
                    #print(mel_batch.shape)
                    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
                audiostyle=self.Audio2style(mel_batch.squeeze(1).squeeze(1))
                wav_sadtalkerexp=torch.tensor(wav_sadtalkerexp)
                code_info, opt_code_dict, cam_info, delta_cam_info = self.build_code_and_cam(wav_sadtalkerexp.cuda(0))

                pred_dict = self.net( "test", self.xy, self.uv,audiostyle.cuda(),  **code_info, **cam_info)
                #input: xy: torch.Size([1, 2, 1024]),   uv:torch.Size([1, 1024, 2]) 
                #code info: appea: torch.Size([1, 127]), shape:torch.Size([1, 179])
                #cam info : batch_Rmats: torch.Size([1, 3, 3])  batch_Tvecs:torch.Size([1, 3, 1])   batch_inv_inmats:torch.Size([1, 3, 3])
                #pred_dict['coarse_dict'] -> dict_keys(['merge_img', 'bg_img']) -> torch.Size([1, 3, 512, 512])

                
                batch_loss_dict = self.loss_utils.calc_total_loss(
                    delta_cam_info=delta_cam_info, opt_code_dict=opt_code_dict, pred_dict=pred_dict,disp_pred_dict=None, 
                    gt_rgb=self.img_tensor, mask_tensor=self.mask_tensor
                )

            optimizer.zero_grad()
            batch_loss_dict["total_loss"].backward()
            optimizer.step()
            scheduler.step()   
            loop_bar.set_description("Opt, Loss: %.6f  " % batch_loss_dict["head_loss"].item())          

            coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
            coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
            # cv2.imwrite("./temp_res/opt_imgs/img_%04d.png" % iter_, coarse_fg_rgb[:, :, ::-1])

        coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
        coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        res_img = np.concatenate([gt_img, coarse_fg_rgb], axis=1)
        
        self.res_img = res_img
        self.res_code_info = code_info
        self.res_cam_info = cam_info


    def save_res(self, base_name, save_root):     
        # Generate Novel Views
        wav_sadtalkerexp=self.sadtalkerexp.generate('/home/dataset/Obama/lipwav/' + '0.jpg', '/home/dataset/Obama/sadtalker/', 'crop', source_image_flag=True, pic_size=256)
        wav_sadtalkerexp=torch.tensor(wav_sadtalkerexp)
        mel_chunks,batch_size,gen=main(0)                
        #print(mel_chunks.shape)
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                                total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            if i == 0:
                model = load_model(wav_checkpoint_path)
                #print ("Model loaded")
    
    
            #print(mel_batch.shape)
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        audiostyle=self.Audio2style(mel_batch.squeeze(1).squeeze(1))
        self.res_code_info, opt_code_dict, cam_info, delta_cam_info = self.build_code_and_cam(wav_sadtalkerexp.cuda())
        render_nv_res = self.render_utils.render_novel_views(self.net, self.res_code_info,audiostyle)
        NVRes_save_path = "%s/FittingResNovelView_%s.gif" % (save_root, base_name)
        #render_nv_res=cv2.cvtColor(render_nv_res, cv2.COLOR_BGR2RGB)
        imageio.mimsave(NVRes_save_path, render_nv_res, 'GIF', duration=self.duration)
        
        # Generate Rendered FittingRes
        img_save_path = "%s/FittingRes_%s.png" % (save_root, base_name)

        self.res_img = put_text_alignmentcenter(self.res_img, self.pred_img_size, "Input", (0,0,0), offset_x=0)
        self.res_img = put_text_alignmentcenter(self.res_img, self.pred_img_size, "Fitting", (0,0,0), offset_x=self.pred_img_size,)

        # self.res_img = cv2.putText(self.res_img, "Input", (110, 240), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
        # self.res_img = cv2.putText(self.res_img, "Fitting", (360, 240), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
        cv2.imwrite(img_save_path, self.res_img[:,:,::-1])

        if self.tar_code_info is not None:
            # "Generate Morphing Res"
            morph_res = self.render_utils.render_morphing_res(self.net, self.res_code_info, self.tar_code_info, self.view_num)
            morph_save_path = "%s/FittingResMorphing_%s.gif" % (save_root, base_name)
            morph_res=cv2.cvtColor(morph_res, cv2.COLOR_BGR2RGB)
            imageio.mimsave(morph_save_path, morph_res, 'GIF', duration=self.duration)

        for k, v in self.res_code_info.items():
            if isinstance(v, torch.Tensor):
                self.res_code_info[k] = v.detach()
        
        temp_dict = {
            "code": self.res_code_info
        }

        torch.save(temp_dict, "%s/LatentCodes_%s_%s.pth" % (save_root, base_name, self.model_name))


    def fitting_single_images(self, img_path, mask_path, para_3dmm_path, tar_code_path, save_root):
        self.load_data(img_path, mask_path, para_3dmm_path)
        base_name = os.path.basename(img_path)[4:-4]

        # load tar code
        if tar_code_path is not None:
            temp_dict = torch.load(tar_code_path, map_location="cpu")
            tar_code_info = temp_dict["code"]
            for k, v in tar_code_info.items():
                if v is not None:
                    tar_code_info[k] = v.to(self.device)
            self.tar_code_info = tar_code_info
        else:
            self.tar_code_info = None
        #tar_code_info is only determining the rendering result of morphing
        self.perform_fitting()
        self.save_res(base_name, save_root)
        
    def _display_current_rendered_image(self,pred_dict,img_tensor):
        coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
        coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        gt_img = (img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        res_img = np.concatenate([gt_img, coarse_fg_rgb], axis=1)


        cv2.imshow('current rendering', res_img)
        cv2.waitKey(0) 
        #closing all open windows 
        cv2.destroyAllWindows() 


if __name__ == "__main__":
    torch.manual_seed(45)  # cpu
    torch.cuda.manual_seed(55)  # gpu
    np.random.seed(65)  # numpy
    random.seed(75)  # random and transforms
    # torch.backends.cudnn.deterministic = True  # cudnn
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser(description='a framework for fitting a single image using HeadNeRF')
    parser.add_argument("--model_path", type=str, required=True)
    
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--para_3dmm", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    
    parser.add_argument("--target_embedding", type=str, default="")
    
    args = parser.parse_args()


    model_path = args.model_path
    save_root = args.save_root
    
    img_path = args.img
    mask_path = args.mask
    para_3dmm_path = args.para_3dmm
    
    if len(args.target_embedding) == 0:
        target_embedding_path = None
    else:
        target_embedding_path = args.target_embedding
    
        temp_str_list = target_embedding_path.split("/")
        if temp_str_list[1] == "*":
            temp_str_list[1] = os.path.basename(model_path)[:-4]
            target_embedding_path = os.path.join(*temp_str_list)
        
        assert os.path.exists(target_embedding_path)
    tt = FittingImage(model_path, save_root, gpu_id=0)
    tt.fitting_single_images(img_path, mask_path, para_3dmm_path, target_embedding_path, save_root)
