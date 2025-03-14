from turtle import color
import torch
from tqdm import tqdm
from HeadNeRFOptions import BaseOptions
import json
import numpy as np
import cv2
import math



import sys
sys.path.insert(1, '/home/colinqian/Project/ETH-XGaze/ETH-XGaze')
try:
    from demo import face_gaze_estimiator,draw_gaze
except:
    pass

from Utils.D6_rotation import gaze_to_d6

class RenderUtils(object):
    def __init__(self, view_num, device, opt: BaseOptions) -> None:
        super().__init__()
        self.view_num = view_num
        self.device = device
        self.opt = opt
        self.build_base_info()
        self.build_cam_info()


    def build_base_info(self):
        mini_h = self.opt.featmap_size
        mini_w = self.opt.featmap_size

        indexs = torch.arange(mini_h * mini_w)
        x_coor = (indexs % mini_w).view(-1)
        y_coor = torch.div(indexs, mini_w, rounding_mode="floor").view(-1)
        
        xy = torch.stack([x_coor, y_coor], dim=0).float()
        uv = torch.stack([x_coor.float() / float(mini_w), y_coor.float() / float(mini_h)], dim=-1)
        
        self.ray_xy = xy.unsqueeze(0).to(self.device)
        self.ray_uv = uv.unsqueeze(0).to(self.device)
        
        with open("ConfigFiles/cam_inmat_info_32x32.json", "r") as f:
            temp_dict = json.load(f)
        # temp_inmat = torch.as_tensor(temp_dict["inmat"])
        temp_inv_inmat = torch.as_tensor(temp_dict["inv_inmat"])
        temp_inv_inmat[:2, :2] /= (self.opt.featmap_size / 32.0)
        self.inv_inmat = temp_inv_inmat.view(1, 3, 3).to(self.device)
        

    def build_cam_info(self):
        tv_z = 0.5 + 11.5
        tv_x = 5.3 

        center_ = np.array([0, 0.0, 0.0]).reshape(3)
        temp_center = np.array([0.0, 0.0, tv_z]).reshape(3)
        temp_cam_center = np.array([[tv_x, 0.0, tv_z]]).reshape(3)

        radius_ = math.sqrt(np.sum((temp_cam_center - center_)**2) - np.sum((temp_center - center_)**2))
        temp_d2 = np.array([[0.0, -1.0, 0.0]]).reshape(3)
        
        cam_info_list = []
        
        angles = np.linspace(0, 360.0, self.view_num)
        for angle in angles:
            theta_ = angle / 180.0 * 3.1415926535
            x_ = math.cos(theta_) * radius_
            y_ = math.sin(theta_) * radius_
            
            temp_vp = np.array([x_, y_, tv_z]).reshape(3)
            d_1 = (center_ - temp_vp).reshape(3)

            d_2 = np.cross(temp_d2, d_1)
            d_3 = np.cross(d_1, d_2)

            d_1 = d_1 / np.linalg.norm(d_1)
            d_2 = d_2 / np.linalg.norm(d_2)
            d_3 = d_3 / np.linalg.norm(d_3)

            rmat = np.zeros((3,3), dtype=np.float32)
            rmat[:, 0] = d_2
            rmat[:, 1] = d_3
            rmat[:, 2] = d_1
            rmat = torch.from_numpy(rmat).view(1, 3, 3).to(self.device)
            tvec = torch.from_numpy(temp_vp).view(1, 3, 1).float().to(self.device)
            
            cam_info = {
                "batch_Rmats": rmat,
                "batch_Tvecs": tvec,
                "batch_inv_inmats": self.inv_inmat,
            }
            cam_info_list.append(cam_info)
        
        base_rmat = torch.eye(3).float().view(1, 3, 3).to(self.device)
        base_rmat[0, 1:, :] *= -1
        base_tvec = torch.zeros(3).float().view(1, 3, 1).float().to(self.device)
        base_tvec[0, 2, 0] = tv_z
        
        self.base_cam_info = {
            "batch_Rmats": base_rmat,
            "batch_Tvecs": base_tvec,
            "batch_inv_inmats": self.inv_inmat,
        }
        
        self.cam_info_list = cam_info_list
        
        
    def render_novel_views(self, net, code_info):
        res_img_list = []
        
        batch_xy = self.ray_xy
        batch_uv = self.ray_uv
        loop_bar = tqdm(range(self.view_num), leave=True)
        for i in loop_bar:
            loop_bar.set_description("Generate Novel Views ")
            cam_info = self.cam_info_list[i]
            with torch.set_grad_enabled(False):
                pred_dict = net("test",batch_xy, batch_uv, **code_info,**cam_info)
            coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
            coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
            
            res_img_list.append(coarse_fg_rgb)
            coarse_fg_rgb = cv2.cvtColor(coarse_fg_rgb, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'/home/lxx/HeadNerf-main-train/test_data/fitting_res/png/{i}.png',coarse_fg_rgb)

        return res_img_list
    
    def render_morphing_res(self, net, code_info_1, code_info_2, nums):
        
        batch_xy = self.ray_xy
        batch_uv = self.ray_uv
        res_img_list = []
        
        loop_bar = tqdm(range(nums), leave=True)
        for i in loop_bar:
            loop_bar.set_description("Generate Morphing Res")
            tv = 1.0 - (i / (nums - 1))
            shape_code = code_info_1["shape_code"] * tv + code_info_2["shape_code"] * (1 - tv)
            appea_code = code_info_1["appea_code"] * tv + code_info_2["appea_code"] * (1 - tv)


            code_info = {
                "bg_code":None,
                "shape_code":shape_code, 
                "appea_code":appea_code
            }
            
            with torch.set_grad_enabled(False):
                pred_dict = net("test",batch_xy, batch_uv, **code_info,**self.base_cam_info)
            coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
            coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
            res_img_list.append(coarse_fg_rgb)
            
        return res_img_list
    
    def calculate_angle_error(self,input_gaze,estimated_gaze):
        error = np.abs(input_gaze-estimated_gaze)
        vertical_error = error[0]
        horizontal_error = error[1]
        return vertical_error,horizontal_error
    

    def render_face_with_gaze(self,net,code_info,face_gaze,scale_factor,gaze_dim,vis_vect=True,cam_info = None):
        batch_xy = self.ray_xy
        batch_uv = self.ray_uv
        shape_code = code_info["shape_code"]
        appea_code = code_info["appea_code"]
        if cam_info is None:
            cam_info = self.base_cam_info
        face_gaze = face_gaze.view(-1)
        face_gaze_feat = face_gaze.repeat(1,gaze_dim//face_gaze.size(0)) * scale_factor

        shape_code[0,-gaze_dim:]= face_gaze_feat

        code_info = {
            "bg_code":None,
            "shape_code":shape_code, 
            "appea_code":appea_code
        }

        with torch.set_grad_enabled(False):
            pred_dict = net("test",batch_xy, batch_uv, **code_info,**cam_info)

        inmat_np = torch.linalg.inv(cam_info['batch_inv_inmats']).detach().cpu().numpy()
        inmat_np = inmat_np.reshape((3,3))
        distortion_np = np.zeros([1,5])
        inmat_np[0,0] *=10; inmat_np[1,1] *=10; inmat_np[0,2] *=10; inmat_np[1,2] *=10
        cam_info = {'camera_matrix':inmat_np, 'camera_distortion':distortion_np}

        coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
        coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
        
        if vis_vect:    
            return self.render_gaze_vect(coarse_fg_rgb.copy(),cam_info,face_gaze)

        return coarse_fg_rgb,0,0

    def render_gaze_vect(self,coarse_fg_rgb,cam_info,face_gaze):
        try:
            face_patch_gaze, pred_gaze_np = face_gaze_estimiator(coarse_fg_rgb.copy(),normalized_input=False,load_self_defined_camera=True,**cam_info)
        except:
            print('no face detected')
            return coarse_fg_rgb, -1, -1
        cv2.putText(img=face_patch_gaze, text=str(pred_gaze_np), org=(0, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 0, 0),thickness=1)

        input_gaze_np = face_gaze.detach().cpu().numpy()
        #input_gaze_np = input_gaze_np * 2 / scale_factor - 1 #map from [0,1] to [-1,1]
        #input_gaze_np = input_gaze_np/np.linalg.norm(input_gaze_np)
        try:
            face_patch_gaze = draw_gaze(face_patch_gaze,input_gaze_np,color=(0,0,255))
        except:
            print(f'error occurs at gaze{input_gaze_np}')
        cv2.putText(img=face_patch_gaze, text=str(input_gaze_np), org=(0, 75), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 255),thickness=1)

        e_v,e_h = self.calculate_angle_error(input_gaze_np,pred_gaze_np)
        cv2.putText(img=face_patch_gaze, text='vertical_e' + str(e_v), org=(0, 450), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 0),thickness=1)
        cv2.putText(img=face_patch_gaze, text='horizontal_e' + str(e_h), org=(0,475), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 0),thickness=1)
        
        return face_patch_gaze,e_v,e_h

    def render_gaze_redirect_res(self, net, code_info_1, code_info_2, nums, scale_factor,gaze_dim,vis_vect=True, D6_rotation=False,cam_info = None):
        ##code1 and code2 only have difference in last few columns (gaze tensor)
        batch_xy = self.ray_xy
        batch_uv = self.ray_uv
        res_img_list = []
        res_imgvec_list = []
        shape_code = code_info_1["shape_code"]
        appea_code = code_info_1["appea_code"]
        if cam_info is None:
            cam_info = self.base_cam_info
        loop_bar = tqdm(range(nums), leave=True)
        e_h_ave = []
        e_v_ave = []
        for i in loop_bar:
            loop_bar.set_description("Generate Morphing Res")
            tv = 1.0 - (i / (nums - 1))
            shape_code = code_info_1["shape_code"]* tv + code_info_2["shape_code"] * (1 - tv)
            appea_code = code_info_1["appea_code"]* tv + code_info_2["appea_code"] * (1 - tv)
            
            gaze_tensor = shape_code[0,-2:]/scale_factor
            if D6_rotation:
                gaze_6d_np = gaze_to_d6(gaze_tensor.cpu().detach().numpy()) * scale_factor
                
                shape_code[0,-gaze_dim:] = torch.from_numpy(gaze_6d_np).repeat(1, gaze_dim//6)

            code_info = {
                "bg_code":None,
                "shape_code":shape_code, 
                "appea_code":appea_code
            }
            
            with torch.set_grad_enabled(False):
                pred_dict = net("test",batch_xy, batch_uv, **code_info,**cam_info)

            inmat_np = torch.linalg.inv(cam_info['batch_inv_inmats']).detach().cpu().numpy()
            inmat_np = inmat_np.reshape((3,3))
            distortion_np = np.zeros([1,5])
            inmat_np[0,0] *=10; inmat_np[1,1] *=10; inmat_np[0,2] *=10; inmat_np[1,2] *=10
            cam_info = {'camera_matrix':inmat_np, 'camera_distortion':distortion_np}
            

            coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
            coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)

            if vis_vect:
                face_patch_gaze, pred_gaze_np = face_gaze_estimiator(coarse_fg_rgb.copy(),normalized_input=False,load_self_defined_camera=True,**cam_info)            
                cv2.putText(img=face_patch_gaze, text=str(pred_gaze_np), org=(0, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 0, 0),thickness=1)
                
                # cv2.imshow('current rendering', face_patch_gaze)
                # cv2.waitKey(0) 
                # #closing all open windows 
                # cv2.destroyAllWindows()  

                input_gaze_np = gaze_tensor.detach().cpu().numpy()
                #input_gaze_np = input_gaze_np * 2 / scale_factor - 1 #map from [0,1] to [-1,1]
                #input_gaze_np = input_gaze_np/np.linalg.norm(input_gaze_np)
                try:
                    face_patch_gaze = draw_gaze(face_patch_gaze,input_gaze_np,color=(0,0,255))
                except:
                    print(f'error occurs at gaze{input_gaze_np}')
                cv2.putText(img=face_patch_gaze, text=str(input_gaze_np), org=(0, 75), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 255),thickness=1)
                
                e_v,e_h = self.calculate_angle_error(input_gaze_np,pred_gaze_np)
                e_v_ave.append(e_v)
                e_h_ave.append(e_h)
                cv2.putText(img=face_patch_gaze, text='vertical_e' + str(e_v), org=(0, 450), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 0),thickness=1)
                cv2.putText(img=face_patch_gaze, text='horizontal_e' + str(e_h), org=(0,475), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 0),thickness=1)

                res_imgvec_list.append(face_patch_gaze)
                face_patch_gaze = cv2.cvtColor(face_patch_gaze, cv2.COLOR_BGR2RGB)
                res_img_list.append(face_patch_gaze)
            else:
                res_imgvec_list.append(coarse_fg_rgb)
                coarse_fg_rgb = cv2.cvtColor(coarse_fg_rgb, cv2.COLOR_BGR2RGB)
                res_img_list.append(coarse_fg_rgb)
                
            
        print(f'Average vertical angle error:{np.mean(e_v_ave)}, Average horizontal angle error:{np.mean(e_h_ave)}')
        return res_img_list,res_imgvec_list