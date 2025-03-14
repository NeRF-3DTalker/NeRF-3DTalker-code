from dis import code_info
import cv2
import torch
import torch.nn.functional as F
import torchvision
import face_alignment


# import lpips
# class VGGPerceptualLoss(torch.nn.Module):
#     def __init__(self, resize=True):
#         super(VGGPerceptualLoss, self).__init__()
#         self.vgg_loss_fn = lpips.LPIPS(net='vgg')


#     def forward(self, input, target,):
#         res = self.vgg_loss_fn(input, target)
#         return res.mean()

import warnings
warnings.filterwarnings('ignore')

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
    

class HeadNeRFLossUtils(object):
    
    def __init__(self, bg_type = "white", use_vgg_loss = True, device = None) -> None:
        super().__init__()
        
        if bg_type == "white":
            self.bg_value = 1.0
        elif bg_type == "black":
            self.bg_value = 0.0
        else:
            self.bg_type = None
            print("Error BG type. ")
            exit(0)
            
        self.use_vgg_loss = use_vgg_loss
        if self.use_vgg_loss:
            assert device is not None
            self.device = device
            self.vgg_loss_func = VGGPerceptualLoss(resize = True).to(self.device)


    @staticmethod
    def calc_cam_loss(delta_cam_info):
        delta_eulur_loss = torch.mean(delta_cam_info["delta_eulur"] * delta_cam_info["delta_eulur"])
        delta_tvec_loss = torch.mean(delta_cam_info["delta_tvec"] * delta_cam_info["delta_tvec"])
        
        return {
            "delta_eular": delta_eulur_loss, 
            "delta_tvec": delta_tvec_loss
        }
        
        
    def calc_code_loss(self, opt_code_dict):
        
        iden_code_opt = opt_code_dict["iden"]
        expr_code_opt = opt_code_dict["expr"]

        iden_loss = torch.mean(iden_code_opt * iden_code_opt)
        expr_loss = torch.mean(expr_code_opt * expr_code_opt)
        
        appea_loss = torch.mean(opt_code_dict["appea"] * opt_code_dict["appea"])

        bg_code = opt_code_dict["bg"]
        if bg_code is None:
            bg_loss = torch.as_tensor(0.0, dtype=iden_loss.dtype, device=iden_loss.device)
        else:
            bg_loss = torch.mean(bg_code * bg_code)
            
        res_dict = {
            "iden_code":iden_loss,
            "expr_code":expr_loss,
            "appea_code":appea_loss,
            "bg_code":bg_loss
        }
        
        return res_dict
    
    
    def calc_data_loss(self, data_dict, gt_rgb, head_mask_c1b, nonhead_mask_c1b):
        
        bg_value = self.bg_value
        
        bg_img = data_dict["bg_img"]
        bg_loss = torch.mean((bg_img - bg_value) * (bg_img - bg_value))
        
        res_img = data_dict["merge_img"]
        res_img = torch.nan_to_num(res_img, nan=0.0)
        head_mask_c3b = head_mask_c1b.expand(-1, 3, -1, -1)
        head_loss = F.mse_loss(res_img[head_mask_c3b], gt_rgb[head_mask_c3b])

        nonhead_mask_c3b = nonhead_mask_c1b.expand(-1, 3, -1, -1)
        temp_tensor = res_img[nonhead_mask_c3b]
        tv = temp_tensor - bg_value
        nonhaed_loss = torch.mean(tv * tv)

        res = {
            "bg_loss": bg_loss,  
            "head_loss": head_loss,  
            "nonhaed_loss": nonhaed_loss,  
        }

        if self.use_vgg_loss:
            masked_gt_img = gt_rgb.clone()
            masked_gt_img[~head_mask_c3b] = bg_value
            
            temp_res_img = res_img
            vgg_loss = self.vgg_loss_func(temp_res_img, masked_gt_img)
            res["vgg"] = vgg_loss

        return res
    
    def calc_disp_loss(self,pred_dict,disp_pred_dict,non_eye_mask_tensor):

        res_img = pred_dict["merge_img"]
        res_img = torch.nan_to_num(res_img, nan=0.0)
        
        disp_img = disp_pred_dict["merge_img"]
        disp_img = torch.nan_to_num(disp_img, nan=0.0)
        
        non_eye_mask_tensor_c3b = non_eye_mask_tensor.expand(-1, 3, -1, -1)

        image_disp_loss = F.l1_loss(res_img[non_eye_mask_tensor_c3b], disp_img[non_eye_mask_tensor_c3b])

        self.fa_func = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        res_img = (res_img* 255).to(dtype = torch.uint8)
        disp_img = (disp_img* 255).to(dtype = torch.uint8)

        lm_disp_loss = torch.tensor(0).to(image_disp_loss.device).float()
        count=0
        for batch_id in range(res_img.size(0)):
            try:
                lm_res = self.fa_func.get_landmarks(res_img[batch_id].permute(1,2,0))[0].float()
                lm_disp = self.fa_func.get_landmarks(disp_img[batch_id].permute(1,2,0))[0].float()
                lm_disp_loss += F.l1_loss(lm_res[:36],lm_disp[:36]) + F.l1_loss(lm_res[48:],lm_disp[48:])
                count+=1
            except:
                pass

        lm_disp_loss/= count + 1e-3


        loss_res = {
            "image_disp_loss":image_disp_loss,
            "lm_disp_loss":lm_disp_loss
        }
  
        return loss_res


    def calc_total_loss(self, delta_cam_info, opt_code_dict, pred_dict, gt_rgb, mask_tensor, disp_pred_dict, eye_mask_tensor=None):
        
        # assert delta_cam_info is not None
        head_mask = (mask_tensor >= 0.5)  
        nonhead_mask = (mask_tensor < 0.5)  

        coarse_data_dict = pred_dict["coarse_dict"]
        loss_dict = self.calc_data_loss(coarse_data_dict, gt_rgb, head_mask, nonhead_mask)
        
        total_loss = 0.0
        for k in loss_dict:
            total_loss += loss_dict[k]
            
        # #cam loss
        # if delta_cam_info is not None:
        #     loss_dict.update(self.calc_cam_loss(delta_cam_info))
        #     total_loss += 0.001 * loss_dict["delta_eular"] + 0.001 * loss_dict["delta_tvec"]
        #
        # # code loss
        # if opt_code_dict is not None:
        #     loss_dict.update(self.calc_code_loss(opt_code_dict))
        #     total_loss += 0.001 * loss_dict["iden_code"] + \
        #                 1.0 * loss_dict["expr_code"] + \
        #                 0.001 * loss_dict["appea_code"] + \
        #                 0.01 * loss_dict["bg_code"]

        #eye mask loss
        # if eye_mask_tensor is not None:
        #     eye_mask = (eye_mask_tensor >= 0.5)
        #     noneye_mask = torch.bitwise_and((eye_mask_tensor < 0.5),head_mask)
        #     loss_dict_eye = self.calc_data_loss(coarse_data_dict, gt_rgb, eye_mask, noneye_mask)
        #     loss_dict['eye_loss'] = loss_dict_eye['head_loss']
        #     total_loss += loss_dict['eye_loss'] * 10
        
        # if disp_pred_dict is not None and eye_mask_tensor is not None:
        #     loss_dict.update(self.calc_disp_loss(pred_dict["coarse_dict"],disp_pred_dict["coarse_dict"],non_eye_mask_tensor=noneye_mask))
        #     total_loss += 3 * loss_dict["image_disp_loss"] + \
        #                   1 * loss_dict["lm_disp_loss"]

        loss_dict["total_loss"] = total_loss
        return loss_dict