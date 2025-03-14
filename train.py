import torch

from talker_trainer import Trainer
from HeadNeRFOptions import get_config,BaseOptions,dataset_config
#from XGaze_utils.data_loader_xgaze_nowavlip import get_data_loader
#from XGaze_utils.data_loader_xgaze import get_data_loader
#from XGaze_utils.data_loader_xgaze_yuan import get_data_loader
from XGaze_utils.data_loader_xgaze_new import get_data_loader
import numpy as np

import configparser
import os
#gpu_list = str([2])
#os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
#print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
def run(config):
    kwargs = {}
    if config.headnerf_options:
        check_dict = torch.load(config.headnerf_options, map_location=torch.device("cpu"))
        para_dict = check_dict["para"]
        opt = BaseOptions(para_dict)
        dataset_config['opt'] =  opt

    if config.use_gpu:
        # ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        np.random.seed(0)
        kwargs = {'num_workers': config.num_workers}

    # instantiate data loaders
    if config.is_train:
        data_loader = get_data_loader(
                        mode='train',
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        dataset_config=dataset_config
                        )
    else:
        data_loader = get_data_loader(
                        mode='test',
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        dataset_config=dataset_config
                        )

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()

if __name__ == '__main__':
    torch.manual_seed(0)
    config, unparsed = get_config()
    run(config)
