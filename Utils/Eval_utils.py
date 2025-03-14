from operator import gt
from skimage.metrics import structural_similarity
import cv2
import numpy as np
from torch.nn import L1Loss
import lpips
import torch

from ignite.metrics import SSIM

def calc_eval_metrics(pred_dict, gt_rgb, mask_tensor,eye_mask_tensor=None,vis=False):
    # head_mask = (mask_tensor >= 0.5)  
    # nonhead_mask = (mask_tensor < 0.5)  

    # coarse_data_dict = pred_dict["coarse_dict"]
    

    # res_img = coarse_data_dict["merge_img"]
    # img_size = res_img.shape[-1]
    # res_img = torch.nan_to_num(res_img, nan=0.0)
    # head_mask_c3b = head_mask.expand(-1, 3, -1, -1)
    # image1 = res_img.view(img_size,img_size,3).cpu().detach().numpy()
    # image2 = gt_rgb.view(img_size,img_size,3).cpu().detach().numpy()

    img_size = mask_tensor.size(-1)
    coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
    coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
    gt_img = (gt_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)

    image1 = coarse_fg_rgb
    image2 = gt_img

    if vis:
        cv2.imshow('image1 rendering', coarse_fg_rgb)
        cv2.waitKey(0) 
        cv2.imshow('image2 rendering', gt_img)
        cv2.waitKey(0) 
        cv2.imshow('mask rendering', mask_tensor.view(img_size,img_size).detach().cpu().numpy())
        cv2.waitKey(0) 
        #closing all open windows 
        cv2.destroyAllWindows() 

    metrics_dict = {
        'SSIM':compute_SSIM_score(image1,image2,vis=vis),
        'PSNR':compute_PSNR_score(image1,image2),
        'LPIPS':compute_LPIPS(image1,image2)
    }
    return metrics_dict





def compute_SSIM_score(image1,image2,vis=False):
    '''
    a perceptual metric that quantifies the image quality degradation
    that is caused by processing such as data compression or by losses in data transmission
    '''
    #Similarity value from 0 to 1
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(image1_gray, image2_gray, full=True)
    #print("Image similarity", score)

    if vis:
        pass
        # diff = (diff * 255).astype("uint8")

        # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = contours[0] if len(contours) == 2 else contours[1]

        # mask = np.zeros(image1.shape, dtype='uint8')
        # filled_after = image2.copy()

        # for c in contours:
        #     area = cv2.contourArea(c)
        #     if area > 40:
        #         x,y,w,h = cv2.boundingRect(c)
        #         cv2.rectangle(image1, (x, y), (x + w, y + h), (36,255,12), 2)
        #         cv2.rectangle(image2, (x, y), (x + w, y + h), (36,255,12), 2)
        #         cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        #         cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

        # cv2.imshow('before', image1)
        # cv2.imshow('after', image2)
        # cv2.imshow('diff',diff)
        # cv2.imshow('mask',mask)
        # cv2.imshow('filled after',filled_after)
        # cv2.waitKey(0)

    return score

def compute_L1_score(image1,image2):
    return np.mean(abs(image1-image2))


def compute_PSNR_score(image1,image2):
    '''
    the ratio between the maximum possible power of an image and 
    the power of corrupting noise that affects the quality of its representation
    '''
    return cv2.PSNR(image1,image2)

def compute_LPIPS(image1,image2):
    loss_fn_alex = lpips.LPIPS(net='alex') 
    h,w,_= image1.shape

    img1 = torch.from_numpy(image1.reshape(-1,3,h,w))
    img2 = torch.from_numpy(image2.reshape(-1,3,h,w))
    d = loss_fn_alex(img1, img2).view(-1).cpu().detach().numpy()
    return d[0]

def compute_PSNR_score_torch(image1,iamge2):
    metric = SSIM(data_range=1.0)
    



if __name__=='__main__':
    before = cv2.imread('XGaze_data/playground/original_image/frame0000_cam00.png')
    after = cv2.imread('XGaze_data/playground/original_image/frame0000_cam01.png')
    score = compute_SSIM_score(before,after)
    score2 = compute_PSNR_score(before,after)
    score3 = compute_LPIPS(before,after)

    print(score)
    print(score2)
    print(score3)