from asyncio import selector_events
from logging import raiseExceptions
#from signal import raise_signal
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from typing import List

import json
import os

import h5py
import cv2
import csv
import pickle as pkl

import sys
sys.path.insert(1,'..')
from XGaze_utils.XGaze_camera_Loader import Camera_Loader
from Utils.D6_rotation import gaze_to_d6

from tqdm import tqdm
import matplotlib.pyplot as plt


# sadtalker
import os

from tqdm import tqdm
import torch
import numpy as np
import random
import scipy.io as scio
import s_utils.audio as audio
#os.environ["CUDA_VISIBLE_DEVICES"]= '2'





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

wav_checkpoint_path='/home/lipwav/pt/checkpoint_step000348000.pth'
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

mel_step_size = 16
device =  'cpu'

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


syncnet_T = 5
syncnet_mel_step_size = 16
from os.path import dirname, join, basename, isfile
from hparams import hparams, get_image_list
from torch import optim

def get_frame_id(frame):
     return int(basename(frame).split('.')[0])

def get_window(start_frame):
    start_id = get_frame_id(start_frame)
    vidname = dirname(start_frame)

    window_fnames = []
    for frame_id in range(start_id, start_id + syncnet_T):
        frame = join(vidname, '{}.jpg'.format(frame_id))
        if not isfile(frame):
            return None
        window_fnames.append(frame)
    return window_fnames

def read_window(window_fnames):
    if window_fnames is None: return None
    window = []
    for fname in window_fnames:
        img = cv2.imread(fname)
        if img is None:
            return None
        try:
            img = cv2.resize(img, (hparams.img_size, hparams.img_size))
        except Exception as e:
            return None

        window.append(img)

    return window

def crop_audio_window(spec, start_frame):
    if type(start_frame) == int:
        start_frame_num = start_frame
    else:
        start_frame_num = get_frame_id(start_frame) # 0-indexing ---> 1-indexing
    start_idx = int(80. * (start_frame_num / float(hparams.fps)))
    
    end_idx = start_idx + syncnet_mel_step_size

    return spec[start_idx : end_idx, :]

def get_segmented_mels(spec, start_frame):
    mels = []
    assert syncnet_T == 5
    start_frame_num = get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
    if start_frame_num - 2 < 0: start_frame_num=3
    for i in range(start_frame_num, start_frame_num + syncnet_T):
        m = crop_audio_window(spec, i - 2)
        if m.shape[0] != syncnet_mel_step_size:
            return None
        mels.append(m.T)

    mels = np.asarray(mels)

    return mels

def prepare_window(window):
    # 3 x T x H x W
    x = np.asarray(window) / 255.
    x = np.transpose(x, (3, 0, 1, 2))

    return x

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    #print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            #print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

def wav_getitem(idx):
    #vidname = self.all_videos[idx]
    img_name = '/home/lipwav/dataroot/dataroot/Obama_/'+str(idx)+'.jpg'
    wrong_img_name = '/home/lipwav/dataroot/dataroot/Obama_/0.jpg'
    window_fnames = get_window(img_name)
    wrong_window_fnames = get_window(wrong_img_name)
    window = read_window(window_fnames)
    wrong_window = read_window(wrong_window_fnames)

    wavpath = join('/home/lipwav/dataroot/dataroot/Obama_/', "audio.wav")
    wav = audio.load_wav(wavpath, hparams.sample_rate)

    orig_mel = audio.melspectrogram(wav).T


    mel = crop_audio_window(orig_mel.copy(), img_name)

    indiv_mels = get_segmented_mels(orig_mel.copy(), img_name)

    window =prepare_window(window)
    y = window.copy()
    window[:, :, window.shape[2]//2:] = 0.

    wrong_window = prepare_window(wrong_window)
    x = np.concatenate([window, wrong_window], axis=0)

    x = torch.FloatTensor(x)
    mel = torch.FloatTensor(mel.T).unsqueeze(0)
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
    y = torch.FloatTensor(y)
    return x, indiv_mels, mel, y








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








def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav

def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames

def generate_blink_seq(num_frames):
    ratio = np.zeros((num_frames,1))
    frame_id = 0
    while frame_id in range(num_frames):
        start = 80
        if frame_id+start+9<=num_frames - 1:
            ratio[frame_id+start:frame_id+start+9, 0] = [0.5,0.6,0.7,0.9,1, 0.9, 0.7,0.6,0.5]
            frame_id = frame_id+start+9
        else:
            break
    return ratio 

def generate_blink_seq_randomly(num_frames):
    ratio = np.zeros((num_frames,1))
    if num_frames<=20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        start = random.choice(range(min(10,num_frames), min(int(num_frames/2), 70))) 
        if frame_id+start+5<=num_frames - 1:
            ratio[frame_id+start:frame_id+start+5, 0] = [0.5, 0.9, 1.0, 0.9, 0.5]
            frame_id = frame_id+start+5
        else:
            break
    return ratio

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













trans_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_train_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'train'
    train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                            transform=trans, is_shuffle=is_shuffle, is_load_label=True)
    #dataloader returns image torch.Size([3, 224, 224]) and label(Gaze)  [2] array
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=False, num_workers=num_workers)

    return train_loader


def get_test_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'test'
    test_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                           transform=trans, is_shuffle=is_shuffle, is_load_label=False)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=num_workers)

    return test_loader

def get_train_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'train'
    train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                            transform=trans, is_shuffle=is_shuffle, is_load_label=True)
    #dataloader returns image torch.Size([3, 224, 224]) and label(Gaze)  [2] array
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=False, num_workers=num_workers)

    return train_loader

def get_data_loader(    mode='train',
                        batch_size=8,
                        num_workers=4,
                        dataset_config=None):

    if dataset_config is None:
        print('dataset configure file required!!')
        raise
    torch.manual_seed(0)
    dataset_config['sub_folder'] = mode #'train' or 'test'
    
    #XGaze_dataset = GazeDataset_normailzed(**dataset_config)
    XGaze_dataset = GazeDataset_normailzed_from_hdf(**dataset_config)

    if mode=='train':
        #training,validation random split
        train_size = int(0.95*len(XGaze_dataset));validation_size = len(XGaze_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(XGaze_dataset, [train_size, validation_size])
        #from sklearn.model_selection import train_test_split
        #train_dataset, val_dataset = train_test_split(XGaze_dataset, test_size=validation_size, shuffle=False)                

        train_loader = DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=True)
        val_loader = DataLoader(val_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=True)

        return (train_loader,val_loader)
    else:
        print('Not implement test dataloader!!')
        raise NotImplementedError
        


#########put this in config file after all testing########################
class BaseOptions(object):
    def __init__(self, para_dict = None) -> None:
        super().__init__()
        
        self.bg_type = "white" # white: white bg, black: black bg.
        
        self.iden_code_dims = 100
        self.expr_code_dims = 79
        self.text_code_dims = 100
        self.illu_code_dims = 27

        self.auxi_shape_code_dims = 179
        self.auxi_appea_code_dims = 127
        
        # self.num_ray_per_img = 972 #972, 1200, 1452, 1728, 2028, 2352
        self.num_sample_coarse = 64
        self.num_sample_fine = 128

        self.world_z1 = 2.5
        self.world_z2 = -3.5
        self.mlp_hidden_nchannels = 384

        if para_dict is None:
            self.featmap_size = 64
            self.featmap_nc = 256       # nc: num_of_channel
            self.pred_img_size = 512
        else:
            self.featmap_size = para_dict["featmap_size"]
            self.featmap_nc = para_dict["featmap_nc"]
            self.pred_img_size = para_dict["pred_img_size"]

################data loader for normalized data from hdf file#############################        
class GazeDataset_normailzed_from_hdf(Dataset):
    def __init__(self, dataset_path: str,
                 opt: BaseOptions,
                 keys_to_use: List[str] = None, 
                 sub_folder='',
                 camera_dir='',
                 _3dmm_data_dir='',
                 transform=None, 
                 is_shuffle=True,
                 index_file=None, 
                 is_load_label=True,
                 device = 'cpu',
                 filter_view=False,
                 gaze_disp = True):
        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label
        self.camera_loader = None#Camera_Loader(camera_dir)
        self._3dmm_data_dir = _3dmm_data_dir
        self.device = device
        self.filter_view = filter_view
        self.gaze_disp = gaze_disp #whether to add gaze displacement 
        if opt is not None:
            self.opt = opt
        else:
            print('option class required, input of opt is None!!')
            raise
        self.img_size = (self.opt.pred_img_size, self.opt.pred_img_size)
        self.pred_img_size = self.opt.pred_img_size
        self.featmap_size = self.opt.featmap_size
        self.featmap_nc = self.opt.featmap_nc
        current_root_path = os.path.split(sys.argv[0])[0]
        sadtalker_paths = init_path('./checkpoints', os.path.join(current_root_path, 's_config'), 256, False, 'crop')
        self.sadtalkerexp = CropAndExtract(sadtalker_paths,'cpu')
        self.batch1,self.batch2,self.batch3,self.batch4 = get_data(os.path.join(self.path, 'aud.wav'), "cpu", ref_eyeblink_coeff_path=None)

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?

        '''
        if self.filter_view:
            ##filter out some severe camera view
            dist_index = [(np.linalg.norm(self.camera_loader[i]['cam_translation']),i) for i in range(18)]
            dist_index.sort()
            self.valid_camera_index = {index for dist,index in dist_index[:10]}#keep camera with 10 least distance

        self.selected_keys = [k for k in keys_to_use] #list of h5 file name
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path,f'processed_{self.selected_keys[num_i]}')
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                hdfs_file = self.hdfs[num_i]
                n = hdfs_file["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)
                                    if hdfs_file['valid_mask'][i] ] #valid frame in subject num_i
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        '''
        self.hdf = None
        self.transform = transform


        

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self, idx):
        #key, idx = self.idx_to_kv[idx]
        mel_chunks,batch_size,gen=main(idx)                
        #print(mel_chunks.shape)
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                                total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            if i == 0:
                model = load_model(wav_checkpoint_path)
                #print ("Model loaded")
    
    
            #print(mel_batch.shape)
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        x, indiv_mels, mel, y=wav_getitem(idx)
        #print(indiv_mels.shape,x.shape) 5 1 80 16  6 5 96 96
        model = Wav2Lip()
        #print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=hparams.initial_learning_rate)
    
        load_checkpoint(wav_checkpoint_path, model, optimizer, reset_optimizer=False)

        g = model(indiv_mels.unsqueeze(0), x.unsqueeze(0))
        #print(g.shape) 1 3 5 96 96
        g = (g[0].detach().cpu().numpy().transpose(1, 2, 3, 0) * 255.).astype(np.uint8)
        cv2.imwrite(os.path.join('/home/temp/' +str(idx)+ '.png'),g[0])
        

        wav_sadtalkerexp=self.sadtalkerexp.generate('/home/temp/' +str(idx)+ '.png', '/home/dataset/Obama/sadtalker/', 'crop', source_image_flag=True, pic_size=256)
        file_path = self.path#os.path.join(self.path,f'processed_{self.selected_keys[key]}')
        #self.hdf = h5py.File(file_path, 'r', swmr=True)
        #assert self.hdf.swmr_mode
        aud_features = np.load(os.path.join(file_path, 'aud.npy'))
        auds=aud_features[min(idx, aud_features.shape[0]-1)]
        auds=np.array(auds).astype(np.float32)


        #print(self.batch3.shape)
        batch={'indiv_mels': self.batch1[:,min(idx, self.batch1.shape[0]-1),:,:,:],  
            # 'ref': ref_coeff, 
            'num_frames': self.batch2, 
            'ratio_gt': self.batch3[:,min(idx, self.batch3.shape[0]-1),:],
            'audio_name': self.batch4}

        img_name = str(idx)+'.jpg'
        mask_name=str(idx)+'_mask'+'.png'
        img_path = os.path.join(file_path,img_name)
        mask_path = os.path.join(file_path,mask_name)
        img_index = idx

        # Get face image
        #<KeysViewHDF5 ['cam_index', 'face_gaze', 'face_head_pose', 'face_mat_norm', 'face_patch',     'frame_index']>
        #               (10098, 1)    (10098, 2)     (10098, 2)     (10098, 3, 3)  (10098, 224, 224, 3)  (10098, 1)
        #                       
        #image = self.hdf['face_patch'][idx, :] ##(224,224,3)
        image = cv2.imread(img_path)
        sadtalkerexp_=self.sadtalkerexp.generate(img_path, '/home/dataset/Obama/sadtalker/', 'crop', source_image_flag=True, pic_size=256)
        sadtalkerexp_0=self.sadtalkerexp.generate('/home/dataset/Obama/png/0.jpg', '/home/dataset/Obama/sadtalker/', 'crop', source_image_flag=True, pic_size=256)
        #print(sadtalkerexp_.shape) (1,64)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#self.hdf['face_patch'][img_index]

        #image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        if self.transform is not None:
            image = self.transform(image)
        image = image.astype(np.float32)/255.0

        self.gt_img_size = image.shape[0]
        if self.gt_img_size != self.pred_img_size:
            image = cv2.resize(image, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

        mask_img =  cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)#self.hdf['mask'][img_index]
        #eye_mask_img = self.hdf['eye_mask'][img_index]
        if mask_img.shape[0] != self.pred_img_size:
            mask_img = cv2.resize(mask_img, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

        #if eye_mask_img.shape[0] != self.pred_img_size:
        #    eye_mask_img = cv2.resize(eye_mask_img, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        

        image[mask_img < 0.5] = 1.0
        img_tensor = (torch.from_numpy(image).permute(2, 0, 1)).unsqueeze(0).to(self.device)#not sure RGB or BRG
        #img_tensor = (torch.from_numpy(image)).unsqueeze(0).to(self.device)#not sure RGB or BRG
        mask_tensor = torch.from_numpy(mask_img[None, :, :]).unsqueeze(0).to(self.device)
        #eye_mask_tensor = torch.from_numpy(eye_mask_img[None, :, :]).unsqueeze(0).to(self.device)

        '''
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][img_index]
            gaze_label = gaze_label.astype('float')
            gaze_tensor = (torch.from_numpy(gaze_label)).to(self.device)
        else:
            gaze_tensor = torch.tensor([None,None])

        gaze_d6 = gaze_to_d6(gaze_label)
        gaze_d6_tensor = (torch.from_numpy(gaze_d6)).to(self.device)

        if self.gaze_disp:
            face_gaze_disp,face_gaze_d6_disp = self.eye_gaze_displacement(gaze_label)

        '''
        #camera_index = self.hdf['cam_index'][img_index][0]

        #camera_parameter = self.camera_loader[camera_index]  ##ground truth camera info

        self.load_3dmm_params(img_index)
        self.dmm_load_3dmm_params(img_index)

        data_info = {'idx':idx,
                        'mel_batch':mel_batch,
                        'wav_gen' :wav_sadtalkerexp,
                        'sad_exp_0' :sadtalkerexp_0,
                        'sad_exp' :sadtalkerexp_,
                        'batch':batch,
                        'auds':auds,
                        'img' : img_tensor,
                        #'gaze': gaze_tensor,  ##only available in training set
                        #'camera_parameter': camera_parameter,
                        '_3dmm': {'cam_info':self.cam_info,
                                  'code_info_i':self.code_info_i,
                                  'code_info':self.code_info},
                        'img_mask' : mask_tensor,
                        #'eye_mask' : eye_mask_tensor,
                        #'gaze_6d' : gaze_d6_tensor,
                        #'gaze_disp' : face_gaze_disp,
                        #'gaze_disp_d6' : face_gaze_d6_disp
                    }
        return data_info

    def load_3dmm_params(self,index):
        # load init codes from the results generated by solving 3DMM rendering opt.
        #nl3dmm_para_dict = self.hdf['nl3dmm']
        
        dmm_name=str(index)+'_nl3dmm.pkl'
        para_3dmm_path = os.path.join(self.path,dmm_name)
        with open(para_3dmm_path, "rb") as f: nl3dmm_para_dict = pkl.load(f)

        base_code = nl3dmm_para_dict["code"].float().detach().unsqueeze(0).to(self.device)
        
        base_iden = base_code[:, :self.opt.iden_code_dims]
        base_expr = base_code[:, self.opt.iden_code_dims:self.opt.iden_code_dims + self.opt.expr_code_dims]
        base_text = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims:self.opt.iden_code_dims 
                                                            + self.opt.expr_code_dims + self.opt.text_code_dims]
        base_illu = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims + self.opt.text_code_dims:]

        self.base_c2w_Rmat =  nl3dmm_para_dict["c2w_Rmat"].float().detach().unsqueeze(0)
        self.base_c2w_Tvec =  nl3dmm_para_dict["c2w_Tvec"].float().detach().unsqueeze(0).unsqueeze(-1)
        self.base_w2c_Rmat =  nl3dmm_para_dict["w2c_Rmat"].float().detach().unsqueeze(0)
        self.base_w2c_Tvec =  nl3dmm_para_dict["w2c_Tvec"].float().detach().unsqueeze(0).unsqueeze(-1)

        temp_inmat =  nl3dmm_para_dict["inmat"].detach().unsqueeze(0)
        temp_inmat[:, :2, :] *= (self.featmap_size / self.gt_img_size)
        
        temp_inv_inmat = torch.zeros_like(temp_inmat)
        temp_inv_inmat[:, 0, 0] = 1.0 / temp_inmat[:, 0, 0]
        temp_inv_inmat[:, 1, 1] = 1.0 / temp_inmat[:, 1, 1]
        temp_inv_inmat[:, 0, 2] = -(temp_inmat[:, 0, 2] / temp_inmat[:, 0, 0])
        temp_inv_inmat[:, 1, 2] = -(temp_inmat[:, 1, 2] / temp_inmat[:, 1, 1])
        temp_inv_inmat[:, 2, 2] = 1.0
        
        #self.temp_inmat = temp_inmat
        self.temp_inv_inmat = temp_inv_inmat

        self.cam_info = {
            "batch_Rmats": self.base_c2w_Rmat.to(self.device),
            "batch_Tvecs": self.base_c2w_Tvec.to(self.device),
            "batch_inv_inmats": self.temp_inv_inmat.to(self.device).float()
        }
        #print(base_expr.shape) (1,79)
        

        self.code_info_i = {
            "base_iden" : base_iden,
            "base_expr" : base_expr,
            "base_text" : base_text,
            "base_illu" : base_illu,
            "inmat" : temp_inmat,
            "inv_inmat" : temp_inv_inmat.float()
        }
        
    
    def dmm_load_3dmm_params(self,index):
        # load init codes from the results generated by solving 3DMM rendering opt.
        #nl3dmm_para_dict = self.hdf['nl3dmm']
        
        dmm_name=str(0)+'_nl3dmm.pkl'
        para_3dmm_path = os.path.join(self.path,dmm_name)
        with open(para_3dmm_path, "rb") as f: nl3dmm_para_dict = pkl.load(f)

        base_code = nl3dmm_para_dict["code"].float().detach().unsqueeze(0).to(self.device)
        
        base_iden = base_code[:, :self.opt.iden_code_dims]
        base_expr = base_code[:, self.opt.iden_code_dims:self.opt.iden_code_dims + self.opt.expr_code_dims]
        base_text = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims:self.opt.iden_code_dims 
                                                            + self.opt.expr_code_dims + self.opt.text_code_dims]
        base_illu = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims + self.opt.text_code_dims:]

        self.base_c2w_Rmat =  nl3dmm_para_dict["c2w_Rmat"].float().detach().unsqueeze(0)
        self.base_c2w_Tvec =  nl3dmm_para_dict["c2w_Tvec"].float().detach().unsqueeze(0).unsqueeze(-1)
        self.base_w2c_Rmat =  nl3dmm_para_dict["w2c_Rmat"].float().detach().unsqueeze(0)
        self.base_w2c_Tvec =  nl3dmm_para_dict["w2c_Tvec"].float().detach().unsqueeze(0).unsqueeze(-1)

        temp_inmat =  nl3dmm_para_dict["inmat"].detach().unsqueeze(0)
        temp_inmat[:, :2, :] *= (self.featmap_size / self.gt_img_size)
        
        temp_inv_inmat = torch.zeros_like(temp_inmat)
        temp_inv_inmat[:, 0, 0] = 1.0 / temp_inmat[:, 0, 0]
        temp_inv_inmat[:, 1, 1] = 1.0 / temp_inmat[:, 1, 1]
        temp_inv_inmat[:, 0, 2] = -(temp_inmat[:, 0, 2] / temp_inmat[:, 0, 0])
        temp_inv_inmat[:, 1, 2] = -(temp_inmat[:, 1, 2] / temp_inmat[:, 1, 1])
        temp_inv_inmat[:, 2, 2] = 1.0
        
        #self.temp_inmat = temp_inmat
        self.temp_inv_inmat = temp_inv_inmat

        '''
        self.cam_info = {
            "batch_Rmats": self.base_c2w_Rmat.to(self.device),
            "batch_Tvecs": self.base_c2w_Tvec.to(self.device),
            "batch_inv_inmats": self.temp_inv_inmat.to(self.device).float()
        }
        '''

        self.code_info = {
            "base_iden" : base_iden,
            "base_expr" : base_expr,
            "base_text" : base_text,
            "base_illu" : base_illu,
            "inmat" : temp_inmat,
            "inv_inmat" : temp_inv_inmat.float()
        }

    def eye_gaze_displacement(self,face_gaze):
        theta = face_gaze[0]; phi = face_gaze[1]
        theta_p = theta + np.random.normal(0 , min(abs(1 - theta) , abs(-1 - theta)))
        phi_p = phi + np.random.normal(0 , min(abs(1 - phi) , abs(-1 - phi)))
        face_gaze_new =  np.array([theta_p,phi_p]).astype('float')
        face_gaze_d6 = gaze_to_d6(face_gaze_new).astype('float')

        face_gaze_disp = (torch.from_numpy(face_gaze_new)).to(self.device)
        face_gaze_d6_disp = (torch.from_numpy(face_gaze_d6)).to(self.device)

        return face_gaze_disp,face_gaze_d6_disp

    def debug_iter(self,idx):
        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path,f'processed_{self.selected_keys[key]}'), 'r', swmr=True)
        assert self.hdf.swmr_mode
        img_index = idx

        image = self.hdf['face_patch'][img_index]

        #image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        if self.transform is not None:
            image = self.transform(image)
        image = image.astype(np.float32)/255.0

        self.gt_img_size = image.shape[0]
        if self.gt_img_size != self.pred_img_size:
            image = cv2.resize(image, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

        mask_img =  self.hdf['mask'][img_index]
        eye_mask_img = self.hdf['eye_mask'][img_index]
        if mask_img.shape[0] != self.pred_img_size:
            mask_img = cv2.resize(mask_img, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

        if eye_mask_img.shape[0] != self.pred_img_size:
            eye_mask_img = cv2.resize(eye_mask_img, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        

        image[mask_img < 0.5] = 1.0
        img_tensor = (torch.from_numpy(image).permute(2, 0, 1)).unsqueeze(0).to(self.device)#not sure RGB or BRG
        #img_tensor = (torch.from_numpy(image)).unsqueeze(0).to(self.device)#not sure RGB or BRG
        mask_tensor = torch.from_numpy(mask_img[None, :, :]).unsqueeze(0).to(self.device)
        eye_mask_tensor = torch.from_numpy(eye_mask_img[None, :, :]).unsqueeze(0).to(self.device)

        
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][img_index]
            gaze_label = gaze_label.astype('float')
            gaze_tensor = (torch.from_numpy(gaze_label)).to(self.device)
        else:
            gaze_tensor = torch.tensor([None,None])

        camera_index = self.hdf['cam_index'][img_index][0]

        camera_parameter = self.camera_loader[camera_index]  ##ground truth camera info

        self.load_3dmm_params(img_index)
        import ipdb
        ipdb.set_trace()
        cv2.imshow('image mask', mask_img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        cv2.imshow('image after masking',image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        cv2.imshow('eye_mask',eye_mask_img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 

################data loader for normalized data#############################
class GazeDataset_normailzed(Dataset):
    def __init__(self, dataset_path: str,
                 opt: BaseOptions,
                 keys_to_use: List[str] = None, 
                 sub_folder='',
                 camera_dir='',
                 _3dmm_data_dir='',
                 transform=None, 
                 is_shuffle=True,
                 index_file=None, 
                 is_load_label=True,
                 device = 'cpu',
                 filter_view=False):
        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label
        self.camera_loader = Camera_Loader(camera_dir)
        self._3dmm_data_dir = _3dmm_data_dir
        self.device = device
        self.filter_view = filter_view
        if opt is not None:
            self.opt = opt
        else:
            print('option class required, input of opt is None!!')
            raise
        self.img_size = (self.opt.pred_img_size, self.opt.pred_img_size)
        self.pred_img_size = self.opt.pred_img_size
        self.featmap_size = self.opt.featmap_size
        self.featmap_nc = self.opt.featmap_nc
        

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        if self.filter_view:
            ##filter out some severe camera view
            dist_index = [(np.linalg.norm(self.camera_loader[i]['cam_translation']),i) for i in range(18)]
            dist_index.sort()
            self.valid_camera_index = {index for dist,index in dist_index[:10]}#keep camera with 10 least distance

        self.selected_keys = [k for k in keys_to_use] #list of h5 file name
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.sub_folder, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                hdfs_file = self.hdfs[num_i]
                n = hdfs_file["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)
                                    if self.is_valid_data_sample(i,hdfs_file)] #our processed image if from 1
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
        self.transform = transform

        #self.debug_iter(0)

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        img_name = str(idx+1).zfill(6)+'.png'
        img_path = os.path.join(self._3dmm_data_dir,img_name)

        # Get face image
        #<KeysViewHDF5 ['cam_index', 'face_gaze', 'face_head_pose', 'face_mat_norm', 'face_patch',     'frame_index']>
        #               (10098, 1)    (10098, 2)     (10098, 2)     (10098, 3, 3)  (10098, 224, 224, 3)  (10098, 1)
        #                       
        #image = self.hdf['face_patch'][idx, :] ##(224,224,3)
        image = cv2.imread(img_path)##(250,250,3)

        #image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        if self.transform is not None:
            image = self.transform(image)
        image = image.astype(np.float32)/255.0

        self.gt_img_size = image.shape[0]
        if self.gt_img_size != self.pred_img_size:
            image = cv2.resize(image, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

        mask_img = cv2.imread(img_path.replace(".png","_mask.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)
        eye_mask_img = cv2.imread(img_path.replace(".png","_mask_eye.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)
        if mask_img.shape[0] != self.pred_img_size:
            mask_img = cv2.resize(mask_img, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

        if eye_mask_img.shape[0] != self.pred_img_size:
            eye_mask_img = cv2.resize(eye_mask_img, dsize=self.img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        

        image[mask_img < 0.5] = 1.0
        img_tensor = (torch.from_numpy(image).permute(2, 0, 1)).unsqueeze(0).to(self.device)#not sure RGB or BRG
        #img_tensor = (torch.from_numpy(image)).unsqueeze(0).to(self.device)#not sure RGB or BRG
        mask_tensor = torch.from_numpy(mask_img[None, :, :]).unsqueeze(0).to(self.device)
        eye_mask_tensor = torch.from_numpy(eye_mask_img[None, :, :]).unsqueeze(0).to(self.device)

        
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][idx, :]
            gaze_label = gaze_label.astype('float')
            gaze_tensor = (torch.from_numpy(gaze_label)).to(self.device)
        else:
            gaze_tensor = torch.tensor([None,None])

        head_pose = self.hdf['face_head_pose'][idx, :]
        head_pose = head_pose.astype('float')
        head_pose = (torch.from_numpy(head_pose)).to(self.device)

        camera_index = self.hdf['cam_index'][idx,:][0]
        camera_parameter = self.camera_loader[camera_index-1]  ##ground truth camera info

        self.load_3dmm_params(os.path.join(self._3dmm_data_dir,img_name.replace(".png","_nl3dmm.pkl")))

        data_info = {
                        'imgname' : img_name,
                        'img_path': img_path,
                        'img' : img_tensor,
                        'gaze': gaze_tensor,  ##only available in training set
                        'head_pose': head_pose,
                        'camera_parameter': camera_parameter,
                        '_3dmm': {'cam_info':self.cam_info,
                                  'code_info':self.code_info},
                        'img_mask' : mask_tensor,
                        'eye_mask' : eye_mask_tensor
                    }
        return data_info

    def is_valid_data_sample(self,i,hdfs_file):
        mm3d_param_exist = os.path.exists(os.path.join(self._3dmm_data_dir,str(i+1).zfill(6) + "_nl3dmm.pkl"))
        
        mask_file = os.path.join(self._3dmm_data_dir,str(i+1).zfill(6) + "_mask.png")
        mask_img = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        valid_mask_img = bool((mask_img>125).any())

        if self.filter_view:
            camera_index  = hdfs_file['cam_index'][i,:]
            is_valid_camera = (camera_index[0] in self.valid_camera_index)
        else:
            is_valid_camera = True


        return mm3d_param_exist & valid_mask_img & is_valid_camera

    def load_3dmm_params(self,para_3dmm_path):
        # load init codes from the results generated by solving 3DMM rendering opt.
        with open(para_3dmm_path, "rb") as f: nl3dmm_para_dict = pkl.load(f)
        base_code = nl3dmm_para_dict["code"].detach().unsqueeze(0).to(self.device)
        
        base_iden = base_code[:, :self.opt.iden_code_dims]
        base_expr = base_code[:, self.opt.iden_code_dims:self.opt.iden_code_dims + self.opt.expr_code_dims]
        base_text = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims:self.opt.iden_code_dims 
                                                            + self.opt.expr_code_dims + self.opt.text_code_dims]
        base_illu = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims + self.opt.text_code_dims:]
        
        self.base_c2w_Rmat = nl3dmm_para_dict["c2w_Rmat"].detach().unsqueeze(0)
        self.base_c2w_Tvec = nl3dmm_para_dict["c2w_Tvec"].detach().unsqueeze(0).unsqueeze(-1)
        self.base_w2c_Rmat = nl3dmm_para_dict["w2c_Rmat"].detach().unsqueeze(0)
        self.base_w2c_Tvec = nl3dmm_para_dict["w2c_Tvec"].detach().unsqueeze(0).unsqueeze(-1)

        temp_inmat = nl3dmm_para_dict["inmat"].detach().unsqueeze(0)
        temp_inmat[:, :2, :] *= (self.featmap_size / self.gt_img_size)
        
        temp_inv_inmat = torch.zeros_like(temp_inmat)
        temp_inv_inmat[:, 0, 0] = 1.0 / temp_inmat[:, 0, 0]
        temp_inv_inmat[:, 1, 1] = 1.0 / temp_inmat[:, 1, 1]
        temp_inv_inmat[:, 0, 2] = -(temp_inmat[:, 0, 2] / temp_inmat[:, 0, 0])
        temp_inv_inmat[:, 1, 2] = -(temp_inmat[:, 1, 2] / temp_inmat[:, 1, 1])
        temp_inv_inmat[:, 2, 2] = 1.0
        
        #self.temp_inmat = temp_inmat
        self.temp_inv_inmat = temp_inv_inmat

        self.cam_info = {
            "batch_Rmats": self.base_c2w_Rmat.to(self.device),
            "batch_Tvecs": self.base_c2w_Tvec.to(self.device),
            "batch_inv_inmats": self.temp_inv_inmat.to(self.device)
        }

        self.code_info = {
            "base_iden" : base_iden,
            "base_expr" : base_expr,
            "base_text" : base_text,
            "base_illu" : base_illu,
            "inmat" : temp_inmat,
            "inv_inmat" : temp_inv_inmat
        }

    def debug_iter(self,idx):
        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        img_name = str(idx+1).zfill(6)+'.png'
        img_path = os.path.join(self._3dmm_data_dir,img_name)

        mask_img = cv2.imread(img_path.replace(".png","_mask.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)
        
        import ipdb 
        ipdb.set_trace()
        self.load_3dmm_params(os.path.join(self._3dmm_data_dir,img_name.replace(".png","_nl3dmm.pkl")))

        # Get face image
        #<KeysViewHDF5 ['cam_index', 'face_gaze', 'face_head_pose', 'face_mat_norm', 'face_patch',     'frame_index']>
        #               (10098, 1)    (10098, 2)     (10098, 2)     (10098, 3, 3)  (10098, 224, 224, 3)  (10098, 1)
        #                       
        image_load = cv2.imread(img_path)

        image_load = image_load[:, :, [2, 1, 0]]  # from BGR to RGB
        image_load = image_load.astype(np.float32)/255.0
        image_load[mask_img < 0.5] = 1.0

        cv2.imshow('image mask', mask_img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        cv2.imshow('image after masking',image_load)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 

def plot_eye_gaze_distribution(dataloader,color='b'):

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    loop_bar = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for idx,data_info in loop_bar:
        gaze = data_info['gaze']
        gaze_np = gaze.view(-1).cpu().detach().numpy()
        plt.scatter(gaze_np[0],gaze_np[1],s=5,c=color)
    
if __name__=='__main__':

    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    # Dataloader = get_data_loader('/home/colinqian/Project/HeadNeRF/headnerf/XGaze_utils/playground',
    #                 '/home/colinqian/Project/HeadNeRF/headnerf/XGaze_utils',
    #                 batch_size=4,num_workers=0)
    # for iter,batch in enumerate(Dataloader):
    #     import ipdb;
    #     ipdb.set_trace()
    #     pass


    #################test normalized data#####################

    from distinctipy import distinctipy
    
    opt = BaseOptions()
    selected_subjects = ['subject0000','subject0003','subject0004','subject0005','subject0006','subject0007','subject0008','subject0009','subject0010','subject0013']
    colors = distinctipy.get_colors(len(selected_subjects))
    import ipdb
    ipdb.set_trace()
    for idx,subject in enumerate(selected_subjects):
        dataset_config={
            'dataset_path': './XGaze_data/processed_data/',
            'opt': BaseOptions(),
            'keys_to_use':[subject], 
            'sub_folder':'train',
            'camera_dir':'./XGaze_data/camera_parameters',
            '_3dmm_data_dir':'./XGaze_data/normalized_250_data',
            'transform':None, 
            'is_shuffle':True,
            'index_file':None, 
            'is_load_label':True,
            'device': 'cpu',
            'filter_view': True
        }
        
        data_loader_train,data_loader_eval = get_data_loader(
                    mode='train',
                    batch_size=1,
                    num_workers=4,
                    dataset_config=dataset_config
                    )
        plot_eye_gaze_distribution(data_loader_train,color=np.array([colors[idx]]))
    plt.show()
    # gaze_dataset = GazeDataset_normailzed(**dataset_config)
    # data_loader = DataLoader(gaze_dataset, batch_size=1, num_workers=4)
    # for iter,batch in enumerate(data_loader):
    #     import ipdb;
    #     ipdb.set_trace()
    #     pass



