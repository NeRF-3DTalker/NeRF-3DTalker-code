# -*- coding: utf-8 -*-
#

import os
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import cv2
from .utils import util
torch.backends.cudnn.benchmark = True
from .utils import lossfunc
from .models.expression_loss import ExpressionLossNet
import torchvision.transforms.functional as F_v
import sys
sys.path.append("external/Visual_Speech_Recognition_for_Multiple_Languages")

def l1_loss(x, y):
    #return torch.mean((x - y) ** 2)
    return torch.mean(torch.abs(x - y))

class Trainer(object):
    def __init__(self, config=None, device='cuda:0'):
        self.cfg = None
        self.device = device


        # deca model
        #self.spectre = model.to(self.device)

        self.global_step = 0


        self.prepare_training_losses()

    def prepare_training_losses(self):



        # ----- initialize lipreader network for lipread loss ----- #
        from external.Visual_Speech_Recognition_for_Multiple_Languages.lipreading.model import Lipreading

        from external.Visual_Speech_Recognition_for_Multiple_Languages.dataloader.transform import Compose, Normalize, CenterCrop, SpeedRate, Identity
        from configparser import ConfigParser
        config = ConfigParser()

        config.read('lipconfigs/lipread_config.ini')
        self.lip_reader = Lipreading(
            config,
            device=self.device
        )

        """ this lipreader is used during evaluation to obtain an estimate of some lip reading metrics
        Note that the lipreader used for evaluation in the paper is different:

        https://github.com/facebookresearch/av_hubert/
        
        to obtain unbiased results
        """

        # ---- initialize values for cropping the face around the mouth for lipread loss ---- #
        # ---- this code is borrowed from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages ---- #
        self._crop_width = 96
        self._crop_height = 96
        self._window_margin = 12
        self._start_idx = 48
        self._stop_idx = 68
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)

        # ---- transform mouths before going into the lipread network for loss ---- #
        self.mouth_transform = Compose([
            Normalize(0.0, 1.0),
            CenterCrop(crop_size),
            Normalize(mean, std),
            Identity()]
        )

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        #print("E flame parameters: ", count_parameters(self.spectre.E_flame))
        #print("flame parameters: ", count_parameters(self.spectre.flame))



    def step(self, img_gen,lmk_gen,img_gt,lmk_gt, phase='train'):
        


        '''
        frame = cv2.imread('/home/lxx/HeadNerf-main-train/linshi/0_.jpg')
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = frame.transpose(2,0,1)
        frame = torch.from_numpy(np.array(frame)).type(dtype = torch.float32).unsqueeze(0)
        #img = F_v.rgb_to_grayscale(frame).squeeze().unsqueeze(0)
        mouths_gt = self.cut_mouth(frame, lmk_gen[0][...,:2].unsqueeze(0))
        #mouths_gt = self.mouth_transform(mouths_gt)
        mouths_gt = mouths_gt.view(-1,1,mouths_gt.shape[-2], mouths_gt.shape[-1])
        cv2.imwrite('/home/lxx/HeadNerf-main-train/' + 'mouth' + '.jpg',(img.detach().cpu().permute(1, 2, 0).numpy()).astype(np.uint8))
        return
        '''
        """ lipread loss - first crop the mouths of the input and rendered faces
        and then calculate the cosine distance of features 
        """
        #print(lmk_gen[...,:2].shape)
        #loss_indices = list(range(2,img_gen.shape[0]-2))
        #for i in range(img_gen.shape[0]):
        #    img_gen[i]=img_gen[i]*255.0
        #    img_gt[i]=img_gt[i]*255.0
        #    img_gen[i] = F_v.rgb_to_grayscale(img_gen[i]).squeeze().unsqueeze(0)
        #    img_gt[i] = F_v.rgb_to_grayscale(img_gt[i]).squeeze().unsqueeze(0)

        #mouths_gt = self.cut_mouth(img_gen, lmk_gen[...,:2])
        #mouths_pred = self.cut_mouth(img_gt, lmk_gt[...,:2])
        #opdict['mouths_gt'] = mouths_gt
        #opdict['mouths_pred'] = mouths_pred
        #print(mouths_gt.shape)
        #cv2.imwrite('/home/lxx/HeadNerf-main-train/' + 'mouth' + '.jpg',(img_gt[0].detach().cpu().permute(1, 2, 0).numpy()).astype(np.uint8))
        #mouths_gt = self.mouth_transform(mouths_gt)
        #mouths_pred = self.mouth_transform(mouths_pred)
        #print(mouths_gt.shape)


        # ---- resize back to BxKx1xHxW (grayscale input for lipread net) ---- #
        #mouths_gt = mouths_gt.view(-1,1,mouths_gt.shape[-2], mouths_gt.shape[-1])
        #mouths_pred = mouths_pred.view(-1,1,mouths_gt.shape[-2], mouths_gt.shape[-1])
        #print(mouths_gt[0].shape)
        



        self.lip_reader.eval()
        self.lip_reader.model.eval()
        
        
        #print(F_v.rgb_to_grayscale(img_gt*255.0).shape)

        lip_features_gt = self.lip_reader.model.encoder(
            F_v.rgb_to_grayscale(img_gt*255.0),
            None,
            extract_resnet_feats=True
        )

        lip_features_pred = self.lip_reader.model.encoder(
            F_v.rgb_to_grayscale(img_gen*255.0),
            None,
            extract_resnet_feats=True
        )

        lip_features_gt = lip_features_gt.view(-1, lip_features_gt.shape[-1])
        lip_features_pred = lip_features_pred.view(-1, lip_features_pred.shape[-1])
        # return l1_loss(lip_features_gt,lip_features_pred)
        # print(images.shape, lip_features_pred.shape)
        lr = (lip_features_gt*lip_features_pred).sum(1)/torch.linalg.norm(lip_features_pred,dim=1)/torch.linalg.norm(lip_features_gt,dim=1)

        losses_lipread = 1-torch.mean(lr)

        return losses_lipread

    def cut_mouth(self, images, landmarks, convert_grayscale=True):
        """ function adapted from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages"""

        mouth_sequence = []

        landmarks = landmarks * 112 + 112
        for frame_idx,frame in enumerate(images):
            window_margin = min(self._window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = landmarks[frame_idx-window_margin:frame_idx + window_margin + 1].mean(dim=0)
            smoothed_landmarks += landmarks[frame_idx].mean(dim=0) - smoothed_landmarks.mean(dim=0)
            #print(smoothed_landmarks.shape)

            center_x, center_y = torch.mean(smoothed_landmarks[self._start_idx:self._stop_idx], dim=0)

            center_x = center_x.round()
            center_y = center_y.round()

            height = self._crop_height//2
            width = self._crop_width//2

            threshold = 5

            if convert_grayscale:
                img = F_v.rgb_to_grayscale(frame).squeeze()
            else:
                img = frame

            if center_y - height < 0:
                center_y = height
            if center_y - height < 0 - threshold:
                raise Exception('too much bias in height')
            if center_x - width < 0:
                center_x = width
            if center_x - width < 0 - threshold:
                raise Exception('too much bias in width')

            if center_y + height > img.shape[-2]:
                center_y = img.shape[-2] - height
            if center_y + height > img.shape[-2] + threshold:
                raise Exception('too much bias in height')
            if center_x + width > img.shape[-1]:
                center_x = img.shape[-1] - width
            if center_x + width > img.shape[-1] + threshold:
                raise Exception('too much bias in width')

            mouth = img[...,int(center_y - height): int(center_y + height),
                                 int(center_x - width): int(center_x + round(width))]

            mouth_sequence.append(mouth)

        mouth_sequence = torch.stack(mouth_sequence,dim=0)
        return mouth_sequence

    def fit(self):
        self.prepare_data()
        start_epoch = 0
        self.global_step = 0

        # initialize outputs close to DECA result (since we find residual from coarse DECA estimate)
        self.spectre.E_expression.layers[0].weight.data *= 0.001
        self.spectre.E_expression.layers[0].bias.data *= 0.001

        self.opt = torch.optim.Adam(
                                self.spectre.E_expression.parameters(),
                                lr=self.cfg.train.lr)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt,[50000],gamma=0.2)

        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            self.epoch = epoch

            all_loss_mean = {}
            for i, batch in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader))):
                if batch is None: continue

                losses, opdict, _ = self.step(batch)

                all_loss = losses['all_loss']

                for key in opdict.keys():
                    opdict[key] = opdict[key].cpu()
                all_loss.backward()

                self.opt.step()
                self.opt.zero_grad()
                # ---- we log the average train loss every 10 steps to obtain a smoother visual curve ---- #
                for key in losses.keys():
                    if key in all_loss_mean:
                        all_loss_mean[key] += losses[key].cpu().item()
                    else:
                        all_loss_mean[key] = losses[key].cpu().item()

                if self.global_step % self.cfg.train.log_steps == 0 and self.global_step > 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {epoch}, Global Iter: {self.global_step}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                    for k, v in all_loss_mean.items():
                        v = v / self.cfg.train.log_steps
                        loss_info = loss_info + f'{k}: {v:.6f}, '
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/'+k, v, global_step=self.global_step)
                    logger.info(loss_info)
                    all_loss_mean = {}

                # ---- visualize several stuff during training ---- #
                if self.global_step % self.cfg.train.vis_steps == 0 and self.global_step > 0:
                    visdict = self.create_grid(opdict)
                    savepath = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir, f'{self.global_step:06}.jpg')
                    util.visualize_grid(visdict, savepath, return_gird=True)

                if self.global_step>0 and self.global_step % self.cfg.train.checkpoint_steps == 0:
                    model_dict = self.spectre.model_dict()
                    model_dict['opt'] = self.opt.state_dict()
                    model_dict['global_step'] = self.global_step
                    model_dict['batch_size'] = self.cfg.dataset.batch_size
                    torch.save(model_dict, os.path.join(self.cfg.output_dir, 'model' + '.tar'))
                    os.makedirs(os.path.join(self.cfg.output_dir, 'models'), exist_ok=True)
                    torch.save(model_dict, os.path.join(self.cfg.output_dir, 'models', f'{self.global_step:08}.tar'))

                # ---- take one random sample of validation and visualize it ---- #
                if self.global_step % self.cfg.train.vis_steps == 0 and self.global_step > 0:
                    for i, eval_batch in enumerate(tqdm(self.val_dataloader)):
                        if eval_batch is None: continue

                        with torch.no_grad():
                            losses, opdict, _ = self.step(eval_batch, phase='val')

                        visdict = self.create_grid(opdict)

                        savepath = os.path.join(self.cfg.output_dir, self.cfg.train.val_vis_dir, f'{self.global_step:08}.jpg')

                        util.visualize_grid(visdict, savepath, return_gird=True)
                        break

                # ---- evaluate the model on the test set every 10k iters ---- #
                if self.global_step % self.cfg.train.evaluation_steps == 0 and self.global_step > 0:
                    self.evaluate(self.test_datasets)

                self.global_step+=1

                scheduler.step()



    def create_grid(self, opdict):
        # Visualize some stuff during training
        shape_images = self.spectre.render.render_shape(opdict['verts'].cuda(), opdict['trans_verts'].cuda())
        input_with_gt_landmarks = util.tensor_vis_landmarks(opdict['images'], opdict['lmk'], isScale=True)

        visdict = {
            'inputs': input_with_gt_landmarks,
            'mouths_gt': opdict['mouths_gt'].unsqueeze(1),
            'mouths_pred': opdict['mouths_pred'].unsqueeze(1),
            'faces_gt': opdict['faces_gt'],
            'faces_pred': opdict['faces_pred'],
            'shape_images': shape_images
        }

        return visdict


    def prepare_data(self):
        from datasets.datasets import get_datasets_LRS3
        self.train_dataset, self.val_dataset, self.test_dataset = get_datasets_LRS3(self.cfg.dataset)

        self.test_datasets = []
        if 'LRS3' in self.cfg.test_datasets:
            self.test_datasets.append(self.test_dataset)

        if 'TCDTIMIT' in self.cfg.test_datasets:
            from datasets.extra_datasets import get_datasets_TCDTIMIT
            _, _, test_dataset_TCDTIMIT = get_datasets_TCDTIMIT(self.cfg.dataset)
            self.test_datasets.append(test_dataset_TCDTIMIT)

        if 'MEAD' in self.cfg.test_datasets:
            from datasets.extra_datasets import get_datasets_MEAD
            _, _, test_dataset_MEAD = get_datasets_MEAD(self.cfg.dataset)
            self.test_datasets.append(test_dataset_MEAD)

        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            if not batch:  # edge case
                return None
            return torch.utils.data.dataloader.default_collate(batch)

        logger.info('---- training data numbers: ', len(self.train_dataset), len(self.val_dataset))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=True,
                            num_workers=self.cfg.dataset.num_workers,
                            pin_memory=True,
                            drop_last=True, collate_fn=collate_fn)

        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False, collate_fn=collate_fn)

