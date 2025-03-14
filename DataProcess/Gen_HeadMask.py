import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from BiSeNet import BiSeNet
from torchvision.transforms import transforms
import cv2
from tqdm import tqdm
import numpy as np
from glob import glob
from correct_head_mask import correct_hair_mask
import argparse


class GenHeadMask(object):
    def __init__(self, gpu_id) -> None:
        super().__init__()
        
        self.device = torch.device("cuda:%s" % gpu_id)
        self.model_path = "ConfigModels/faceparsing_model.pth"
        
        self.init_model()
        self.lut = np.zeros((256, ), dtype=np.uint8)
        #self.lut[1:14] = 1 
        self.lut[1:14] = 1
        self.lut[17] = 2 #hair
        
        #  ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    def init_model(self):
        n_classes = 19
        net = BiSeNet(n_classes=n_classes).to(self.device)
        net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net = net
        self.net.eval()

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    def green_screen_filtering(self,image,cur_mask):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        u_green = np.array([70, 255, 255])
        l_green = np.array([36, 0, 0])
        mask = cv2.inRange(hsv, l_green, u_green)
        mask = cv2.bitwise_not(mask)
        target = cv2.bitwise_and(cur_mask,cur_mask, mask=mask)
        return target

    def _extend_eye_mask(self,mask,size=(20,20)):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
        dilated_mask = cv2.dilate(mask, kernel, iterations=3)
        return dilated_mask

    def extrace_eye_mask(self,parsing,mask_extend_size=(20,20)):
        eye_mask = np.zeros((parsing.shape[0],parsing.shape[1]))
        for pi in [4,5]:#parsing index for eye
            index = np.where(parsing == pi)
            eye_mask[index[0], index[1]] = 255
        eye_mask = eye_mask.astype(np.uint8)

        eye_mask = self._extend_eye_mask(eye_mask)

        return eye_mask



    def main_process(self, img_dir,*args):
        img_path_list = [x for x in glob("%s/*.jpg" % img_dir) if "mask" not in x]  
        if len(img_path_list) == 0:
            print("Dir: %s does include any .png images." % img_dir)
            exit(0)
        img_path_list.sort()
        loop_bar = tqdm(img_path_list)
        loop_bar.set_description("Generate head masks")
        for img_path in loop_bar:
            save_path = img_path[:-4] + "_mask.png"
            eye_mask_path = img_path[:-4] + "_mask_eye.png"
            img = cv2.imread(img_path)
            width,height,_ = img.shape
            vis_image = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(512,512))  #resize for better segmentation of eyes
            temp_image =img.copy()
            img = self.to_tensor(img)
            img = img.unsqueeze(0)
            img = img.to(self.device)
            with torch.set_grad_enabled(False):
                pred_res = self.net(img)
                out = pred_res[2]  ##select the finetuned degree of segmentation results

            res = out.squeeze(0).cpu().numpy().argmax(0)
            res = res.astype(np.uint8)
            eye_mask = self.extrace_eye_mask(res)#extract eye segmentation
            cv2.LUT(res, self.lut, res)
            
            res = correct_hair_mask(res)

            res[res != 0] = 255
            res = self.green_screen_filtering(temp_image,res)

            eye_mask = np.bitwise_and(res,eye_mask)#filter out infeasible eye seg

            eye_mask = cv2.resize(eye_mask,(width,height))
            res = cv2.resize(res,(width,height))


            # cv2.imshow('current mask', vis_image)
            # cv2.waitKey(0) 
            # vis_image[res!=255] = 0
            # cv2.imshow('masked image', res)
            # cv2.waitKey(0) 
            # vis_image[eye_mask!=255] = 0
            # cv2.imshow('masked image', eye_mask)
            # cv2.waitKey(0) 

            # cv2.destroyAllWindows() 
            # temp_img = bgr_img.copy()
            # temp_img[res == 0] = 255
            
            # res = np.concatenate([bgr_img, temp_img], axis=1)
            cv2.imwrite(save_path, res)
            cv2.imwrite(eye_mask_path, eye_mask)
            
    def process_single_image(self,img):
        '''
        input image
        output masked image and eye mask
        '''

        width,height,_ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(512,512))  #resize for better segmentation of eyes
        temp_image =img.copy()
        img = self.to_tensor(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        with torch.set_grad_enabled(False):
            pred_res = self.net(img)
            out = pred_res[2]  ##select the finetuned degree of segmentation results

        res = out.squeeze(0).cpu().numpy().argmax(0)
        res = res.astype(np.uint8)
        eye_mask = self.extrace_eye_mask(res)#extract eye segmentation
        cv2.LUT(res, self.lut, res)
        
        res = correct_hair_mask(res)

        res[res != 0] = 255
        res = self.green_screen_filtering(temp_image,res)

        eye_mask = np.bitwise_and(res,eye_mask)#filter out infeasible eye seg

        eye_mask = cv2.resize(eye_mask,(width,height))
        res = cv2.resize(res,(width,height))

        return (res,eye_mask)





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='The code for generating head mask images.')
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--format", type=str, default='png')
    args = parser.parse_args()

    gpu_id = args.gpu_id
    img_dir = args.img_dir

    # assert len(sys.argv) == 2
    # img_dir = sys.argv[1]

    tt = GenHeadMask(gpu_id=gpu_id)
    tt.main_process(img_dir=img_dir)
    