#!/usr/bin/env python
import os
import torch
import numpy as np
import cv2

from base.utilities import get_parser, get_logger
from models import get_model
from base.baseTrainer import load_state_dict

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

cfg = get_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.test_gpu)
import torch.nn as nn

class classifier_AU(nn.Module):
    def __init__(self):
        super(classifier_AU, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=1), nn.InstanceNorm2d(16),nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, stride=1, padding=1), nn.InstanceNorm2d(32),nn.LeakyReLU(0.2, True))
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 5, stride=2, padding=1), nn.InstanceNorm2d(64),nn.LeakyReLU(0.2, True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=1, padding=1), nn.InstanceNorm2d(128),nn.LeakyReLU(0.2, True))
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.linear0  = nn.Sequential(nn.Linear(3200 ,1600), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        self.linear1  = nn.Sequential(nn.Linear(1600 ,512), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        self.linear2  = nn.Sequential(nn.Linear(512 ,256), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        self.linear3  = nn.Sequential(nn.Linear(256 ,64), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        self.linear4 = nn.Linear(64 ,1)
        #self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        #print(inputs.shape)
        out = self.conv1(inputs)       
        out = self.conv2(out)
        out = self.maxpool1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.maxpool2(out)
        #print(out.shape)
        out = out.contiguous().view(out.shape[0], -1)
        out = self.linear0(out)
        out = self.linear1(out)
        linear2_out = self.linear2(out)
        linear3_out = self.linear3(linear2_out)
        out = self.linear4(linear3_out)
        out = self.sigmoid(out)
        #out = self.softmax(out)
        return linear2_out,out
def main():
    global cfg, logger
    logger = get_logger()
    logger.info(cfg)
    logger.info("=> creating model ...")
    model = get_model(cfg)
    model = model.cuda()
    classifier_AU10=classifier_AU()
    classifier_AU14=classifier_AU()
    classifier_AU20=classifier_AU()
    classifier_AU25=classifier_AU()
    classifier_AU26=classifier_AU()
    if os.path.isfile(cfg.model_path):
        logger.info("=> loading checkpoint '{}'".format(cfg.model_path))
        checkpoint = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cpu())
        load_state_dict(model, checkpoint['state_dict'])
        logger.info("=> loaded checkpoint '{}'".format(cfg.model_path))
    else:
        raise RuntimeError("=> no checkpoint flound at '{}'".format(cfg.model_path))
    if os.path.isfile('/home/RUN/vocaset/CodeTalker_s1/classifier_model10/model.pth.tar'):
        logger.info("=> loading checkpoint '{}'".format('/home/RUN/vocaset/CodeTalker_s1/classifier_model10/model.pth.tar'))
        checkpoint = torch.load('/home/RUN/vocaset/CodeTalker_s1/classifier_model10/model.pth.tar', map_location=lambda storage, loc: storage.cpu())
        load_state_dict(classifier_AU10, checkpoint['state_dict'])
    if os.path.isfile('/home/RUN/vocaset/CodeTalker_s1/classifier_model14/model.pth.tar'):
        logger.info("=> loading checkpoint '{}'".format('/home/RUN/vocaset/CodeTalker_s1/classifier_model14/model.pth.tar'))
        checkpoint = torch.load('/home/RUN/vocaset/CodeTalker_s1/classifier_model14/model.pth.tar', map_location=lambda storage, loc: storage.cpu())
        load_state_dict(classifier_AU14, checkpoint['state_dict'])
    if os.path.isfile('/home/RUN/vocaset/CodeTalker_s1/classifier_model20/model.pth.tar'):
        logger.info("=> loading checkpoint '{}'".format('/home/RUN/vocaset/CodeTalker_s1/classifier_model20/model.pth.tar'))
        checkpoint = torch.load('/home/RUN/vocaset/CodeTalker_s1/classifier_model20/model.pth.tar', map_location=lambda storage, loc: storage.cpu())
        load_state_dict(classifier_AU20, checkpoint['state_dict'])
    if os.path.isfile('/home/RUN/vocaset/CodeTalker_s1/classifier_model25/model.pth.tar'):
        logger.info("=> loading checkpoint '{}'".format('/home/RUN/vocaset/CodeTalker_s1/classifier_model25/model.pth.tar'))
        checkpoint = torch.load('/home/RUN/vocaset/CodeTalker_s1/classifier_model25/model.pth.tar', map_location=lambda storage, loc: storage.cpu())
        load_state_dict(classifier_AU25, checkpoint['state_dict'])
    if os.path.isfile('/home/RUN/vocaset/CodeTalker_s1/classifier_model26/model.pth.tar'):
        logger.info("=> loading checkpoint '{}'".format('/home/RUN/vocaset/CodeTalker_s1/classifier_model26/model.pth.tar'))
        checkpoint = torch.load('/home/RUN/vocaset/CodeTalker_s1/classifier_model26/model.pth.tar', map_location=lambda storage, loc: storage.cpu())
        load_state_dict(classifier_AU26, checkpoint['state_dict'])
        
    # ####################### Data Loader ####################### #
    from dataset.data_loader_test import get_dataloaders
    dataset = get_dataloaders(cfg)
    test_loader = dataset['valid']

    test(model, test_loader,classifier_AU10,classifier_AU14,classifier_AU20,classifier_AU25,classifier_AU26, save=True)



def test(model, test_loader,classifier_AU10,classifier_AU14,classifier_AU20,classifier_AU25,classifier_AU26, save=False):
    model.eval()
    save_folder = os.path.join(cfg.save_folder, 'npy')
    #if not os.path.exists(save_folder):
    #    os.makedirs(save_folder)

    with torch.no_grad():
        for i, (data,mask,idx,data_) in enumerate(test_loader):
            data = data.cuda()#cuda(non_blocking=True)
            #template = template.cuda(non_blocking=True)
            data=data.squeeze(0)
            
            data_=data_.cuda()
            data_ = data_.squeeze(0)
            
            classifier_AU10=classifier_AU10.cuda()
            classifier_AU10=classifier_AU14.cuda()
            classifier_AU10=classifier_AU20.cuda()
            classifier_AU10=classifier_AU25.cuda()
            classifier_AU10=classifier_AU26.cuda()
            linear2_au10,predict_au10=classifier_AU10(data_)
            linear2_au14,predict_au14=classifier_AU14(data_)
            linear2_au20,predict_au20=classifier_AU20(data_)
            linear2_au25,predict_au25=classifier_AU25(data_)
            linear2_au26,predict_au26=classifier_AU26(data_)
            AU_fea=torch.cat((linear2_au10.repeat(40, 1), linear2_au14.repeat(40, 1),linear2_au20.repeat(40, 1),linear2_au25.repeat(40, 1),linear2_au26.repeat(40, 1)), dim=0)
            out, _, _ = model(data,AU_fea.unsqueeze(0))
            #print(out.shape)
            #out[1,:,:]=data.squeeze(0)[1,:,:]
            #out[2,:,:]=data.squeeze(0)[2,:,:]
            #out[2]=data[0][2]
            #out[1]=data[0][1]
            out.permute(1, 2, 0)[mask[0]<0.5]=1.0

            
            coarse_fg_rgb_ = (out.detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
            res_img=cv2.cvtColor(coarse_fg_rgb_, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join('/home/dataset_jiaqiang/Obama/gen+/',str(idx.item()) + '.png'),res_img)
            #out = out.squeeze()

            #if save:
            #    np.save(os.path.join(save_folder, file_name[0].split(".")[0]+".npy"), out.detach().cpu().numpy())


if __name__ == '__main__':
    main()
