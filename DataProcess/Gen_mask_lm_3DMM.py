from pyclbr import Class
from re import sub
import h5py
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import cv2
import h5py
import argparse
import torch

from Gen_HeadMask import GenHeadMask
from Gen_Landmark import  Gen2DLandmarks
import sys
sys.path.append("/home/lxx/HeadNerf-main-train/Fitting3DMM/")

#sys.path.append("/local/home/chqian/HeadNerf/Fitting3DMM")

from FittingNL3DMM import FittingNL3DMM_from_h5py


nl3d_param_shape = {
    "code": (306,),
    "w2c_Rmat":(3,3),
    "w2c_Tvec":(3,),
    "inmat":(3,3),
    "c2w_Rmat":(3,3),
    "c2w_Tvec":(3,),
    "inv_inmat":(3,3)
}

class Data_Processor(object):
    def __init__(self,img_dir,save_dir,img_size, intermediate_size,hdf_file=False):
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.img_size = img_size
        self.intermediate_size = intermediate_size
        if hdf_file:
            self.hdf_input_dir = img_dir
        else:
            self.hdf_input_dir = None
        self.load_utils()

    def generate_head_mask(self,image):
        head_mask,eye_mask = self.headmask_generator.process_single_image(image)
        return head_mask,eye_mask

    def generate_landmarks(self,image):
        landmarks = self.landmark_generator.process_single_image(image)

        return landmarks  #(136,) numpy

    def generate_nl3dmm_parameter(self,image,lm_info):
        nl_param = self.NL3DMM_param_generator.process_single_image(image,lm_info)
        return nl_param

    def load_utils(self,gpu_id=0):
        self.headmask_generator = GenHeadMask(gpu_id=gpu_id)
        self.landmark_generator = Gen2DLandmarks()
        self.NL3DMM_param_generator = FittingNL3DMM_from_h5py(self.img_size,self.intermediate_size,gpu_id=gpu_id)

    
    def is_valid_image_path(self,image_path):
        mask_not_exist = "mask" not in os.path.basename(image_path) #not mask png file
        lm_not_exist = not os.path.exists(image_path.replace(".png","_lm2d.txt")) #landmark file not exist

        return mask_not_exist & lm_not_exist


    def load_image(self,image_idx,image_dir):
        image_path = os.path.join(image_dir,str(image_idx).zfill(6)+'.png')
        return cv2.imread(image_path)

    def load_lm2d(self,image_idx,image_dir):
        lm_path = os.path.join(image_dir,str(image_idx).zfill(6)+'_lm2d.txt')
        return np.loadtxt(lm_path, dtype = np.float32)

    def process_data_from_hdf_file(self,sub_id,image_patch_size=250):
        

        if self.hdf_input_dir is None:
            print('Set hdf file to True to enable loading from hdf file!!')

        input_file = os.path.join(self.hdf_input_dir , 'subject' +str(sub_id).zfill(4) + '.h5')

        if not os.path.exists(input_file):
            print(f'Current subject {sub_id} not exist!!')
            return

        with h5py.File(os.path.join(self.save_dir,f'processed_subject{str(sub_id).zfill(4)}'),'w') as fid,h5py.File(input_file,'r') as src_file:
            
            self.include_gaze = False
            num_data = src_file.attrs['data_size']

            try:
                face_patches = src_file['face_patch']
            except:
                print('face patch not found!!')
                fid.create_dataset("valid_mask", data = [False]*num_data)##if num data is zero, valid mask should be []
                return

            if 'face_gaze' in src_file.keys():
                self.include_gaze = True
                gazes = src_file['face_gaze']
            frames_indexs = src_file['frame_index']
            cam_indexes = src_file['cam_index']

            if 'mask' not in fid.keys():
                output_face_mask= fid.create_dataset("mask", shape=(num_data, image_patch_size, image_patch_size),
                                                                        compression='lzf', dtype=np.uint8,
                                                                        chunks=(1, image_patch_size, image_patch_size))
                output_eye_mask= fid.create_dataset("eye_mask", shape=(num_data, image_patch_size, image_patch_size),
                                                                        compression='lzf', dtype=np.uint8,
                                                                        chunks=(1, image_patch_size, image_patch_size))
                output_lm2d= fid.create_dataset("lm2d", shape=(num_data, 136),
                                                                        dtype=float,chunks=(1, 136))
                nl3d_param_output={}
                for key,shape in nl3d_param_shape.items():
                    nl3d_param_output[key]= fid.create_dataset(f"nl3dmm/{key}", shape=(num_data, *shape),
                                                            dtype=float,chunks=(1, *shape))

                output_frame_index = fid.create_dataset("frame_index", shape=(num_data, 1),
                                                dtype=int, chunks=(1, 1),data = frames_indexs[:num_data])
                output_cam_index = fid.create_dataset("cam_index", shape=(num_data, 1),
                                                    dtype=int, chunks=(1, 1), data = cam_indexes[:num_data])
                if self.include_gaze:   
                    output_face_gaze = fid.create_dataset("face_gaze", shape=(num_data, 2),
                                                    dtype=float, chunks=(1, 2), data = gazes[:num_data,:])

                output_face_patches= fid.create_dataset("face_patch", shape=(num_data, image_patch_size, image_patch_size,3),
                                                                        compression='lzf', dtype=np.uint8,
                                                                       chunks=(1, image_patch_size, image_patch_size,3))

            valid_mask = [False] * num_data
            for num_i in tqdm(range(num_data)):
                face_patch = face_patches[num_i, :]  # the face patch
                # if 'face_gaze' in fid.keys():
                #     gaze = fid['face_gaze'][num_i, :]   # the normalized gaze direction with size of 2 dimensions as horizontal and vertical gaze directions.
                # frame_index = fid['frame_index'][num_i, 0]  # the frame index
                # cam_index = fid['cam_index'][num_i, 0]   # the camera index     
                face_patch = cv2.resize(face_patch, (image_patch_size, image_patch_size))   

                mask, eye_mask, lm2d, nl3dmm_param = self.process_mask_and_landmark_for_single_image(face_patch)

                # face_patch_loaded = self.load_image(num_i,'test_data/playground')
                # lm_loaded = self.load_lm2d(num_i,'test_data/playground')


                output_face_patches[num_i] = face_patch
                output_face_mask[num_i] = mask
                output_eye_mask[num_i] = eye_mask
                output_lm2d[num_i] = lm2d
                if nl3dmm_param is not None:
                    valid_mask[num_i] = True
                    for key in nl3d_param_output.keys():
                        nl3d_param_output[key][num_i] = nl3dmm_param[key]
                if np.isnan(nl3d_param_output['code'][num_i]).any():
                    ##nan detected
                    print('detect nan in nl3d parameters')
                    valid_mask[num_i] = False
                
                    # import ipdb
                    # ipdb.set_trace()
                    # cv2.imshow('current mask', mask)
                    # cv2.waitKey(0) 
                    # cv2.imshow('masked image', eye_mask)
                    # cv2.waitKey(0) 
                    # face_patch[mask!=255] = 0
                    # cv2.imshow('masked image', face_patch)
                    # cv2.waitKey(0) 
                    # face_patch[eye_mask!=255] = 0
                    # cv2.imshow('masked image', face_patch)
                    # cv2.waitKey(0) 
                    # cv2.destroyAllWindows() 




            fid.create_dataset("valid_mask", data = valid_mask)##if num data is zero, valid mask should be []
        
            

    def process_mask_and_landmark_for_single_image(self,img):
        mask,eye_mask = self.generate_head_mask(img)

        lm2d = self.generate_landmarks(img)

        if lm2d is not None:
            nl3dmm_param = self.generate_nl3dmm_parameter(img.copy(),lm2d.copy())
        else:
            nl3dmm_param = None
        
        return mask,eye_mask,lm2d,nl3dmm_param



    def process_mask_and_landmark(self,subject_id):
        self.load_utils()
        h5py_filename = str(subject_id).zfill(5) + ".hdf5"

        img_path_list = [x for x in glob("%s/*.png" % self.img_dir) if self.is_valid_image_path(x)]

        if len(img_path_list) == 0:
            print("Dir: %s does include any .png images." % self.img_dir)
            exit(0)

        img_path_list.sort()
        with h5py.File(os.path.join(self.save_dir,h5py_filename), "w") as file:
            for idx,img_path in tqdm(enumerate(img_path_list), desc="Generate facial mask and landmarks"): 
                
                img = cv2.imread(img_path)

                mask,eye_mask = self.generate_head_mask(img)

                lm2d = self.generate_landmarks(img)

                img_name = "subject" + str(subject_id).zfill(5) + f"_{idx}"

                file.create_dataset(f'{img_name}/image', data = img)
                file.create_dataset(f'{img_name}/mask', data = mask)
                file.create_dataset(f'{img_name}/eye_mask', data = eye_mask)
                file.create_dataset(f'{img_name}/lm2d', data = lm2d)

                if lm2d is not None:
                    nl3dmm_param = self.generate_nl3dmm_parameter(img,lm2d)
                    for k,v in nl3dmm_param.items():
                        file.create_dataset(f'{img_name}/nl3d_param/{k}', data = v)

                file[img_name].attrs['id'] = idx
                file[img_name].attrs['size'] = img.shape[0]
    




def load_h5py_file(file_path):
     with h5py.File(os.path.join(file_path), "r") as file:
         for f in file:
             print(f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='The code for generating facial landmarks and head mask')
    # parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--sb", type=int, required=True,default=0)
    parser.add_argument("--se", type=int, required=True,default=1)
    args = parser.parse_args()

    torch.cuda.set_device(3)
    test = Data_Processor(args.img_dir,args.save_dir,250,125,hdf_file=True)
    for sub_id in range(args.sb,args.se):
        test.process_data_from_hdf_file(sub_id)
    #test.process_mask_and_landmark(0)
    #load_h5py_file(args.save_dir + '/00000.hdf5')



                

        
    

