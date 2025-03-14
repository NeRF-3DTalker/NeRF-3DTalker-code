import json
import os
import argparse

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
            self.featmap_size = 32
            self.featmap_nc = 256       # nc: num_of_channel
            self.pred_img_size = 256
        else:
            self.featmap_size = para_dict["featmap_size"]
            self.featmap_nc = para_dict["featmap_nc"]
            self.pred_img_size = para_dict["pred_img_size"]


dataset_config={
    'dataset_path': '/home/lxx/HeadNerf-main-train/dataset/C0/png',
    'opt': BaseOptions(),
    'keys_to_use':['subject0000','subject0003','subject0004','subject0005','subject0006','subject0007','subject0008','subject0009','subject0010','subject0013'], 
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


arg_lists = []
parser = argparse.ArgumentParser(description='RAM')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--data_dir', type=str, default='data/xgaze',
                      help='Directory of the data')
data_arg.add_argument('--batch_size', type=int, default=1,
                      help='# of images in each batch of data')         
data_arg.add_argument('--num_workers', type=int, default=5,
                      help='# of subprocesses to use for data loading')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--epochs', type=int, default=30,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=0.0001,
                       help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=10,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--lr_decay_factor', type=float, default=0.1,
                       help='Number of epochs to wait before reducing lr')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--gpu_id', type=int,default=1,
                      help='Which gpu to use')                    
misc_arg.add_argument('--pre_trained_model_path', type=str, default='./ckpt/epoch_24_ckpt.pth.tar',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--print_freq', type=int, default=1,
                      help='How frequently to print training details')
misc_arg.add_argument('--ckpt_dir', type=str, default='./logs/ckpt',
                      help='Directory in which to save model checkpoints')

misc_arg.add_argument('--headnerf_options', type=str, default='',
                      help='File path that can load headnerf options (Baseoption class) and model parameters')

misc_arg.add_argument('--use_gt_camera', type=str2bool, default=False,
                      help="Whether use gt camera parameter in ETH_XGaze")

misc_arg.add_argument('--include_eye_gaze', type=str2bool, default=False,
                    help="Whether to include eye gaze as additional input to headnerf")
misc_arg.add_argument('--eye_gaze_dimension', type=int, default=2,
                      help='Decide to which dimension we would like eye gaze input to extend')
misc_arg.add_argument('--eye_gaze_scale_factor', type=int, default=1,
                      help='Decide to how much you want to scale normalized face gaze')
misc_arg.add_argument('--eye_gaze_disentangle', type=str2bool, default=True,
                      help='Decide to whether to use disentangle module for eye gaze (high computational cost)')
misc_arg.add_argument('--gaze_D6_rotation', type=str2bool, default=False,
                    help="Whether to use 6D rotation representation instead of [-1,1] gaze")


misc_arg.add_argument('--resume', type=str, default='',
                      help='whether to resume training, load saved ckpt')        

misc_arg.add_argument('--comment', type=str, default='',
                      help='use comment to describ experiment')                

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
