#!/usr/bin/env python
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
import cv2

from base.baseTrainer import poly_learning_rate, reduce_tensor, save_checkpoint
from base.utilities import get_parser, get_logger, main_process, AverageMeter
from models import get_model
from metrics.loss import calc_vq_loss
from torch.optim.lr_scheduler import StepLR
#gpu_list = str([2])
#os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
#print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.train_gpu = args.train_gpu[0]
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)
def load_ckpt(model, ckpt_path):
    old_state_dict = ckpt_path
    cur_state_dict = model.state_dict()
    for param in cur_state_dict:

        old_param = param
        if old_param in old_state_dict and cur_state_dict[param].size()==old_state_dict[old_param].size():
            print("loading param: ", param)
            model.state_dict()[param].data.copy_(old_state_dict[old_param].data)
        else:
            print("warning cannot load param: ", param)
from base.baseTrainer import load_state_dict
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
        return linear3_out,out

def main_worker(gpu, ngpus_per_node, args):
    cfg = args
    cfg.gpu = gpu

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size,
                                rank=cfg.rank)
    # ####################### Model ####################### #
    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(cfg.save_path)
    model = get_model(cfg)
    if cfg.sync_bn:
        logger.info("using DDP synced BN")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if main_process(cfg):
        logger.info(cfg)
        logger.info("=> creating model ...")
        # model.summary(logger, writer)
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
        cfg.batch_size_val = int(cfg.batch_size_val / ngpus_per_node)
        cfg.workers = int(cfg.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(gpu), device_ids=[gpu])
    else:
        torch.cuda.set_device(gpu)
        
    

    
    # ####################### Optimizer ####################### #



    # ####################### Data Loader ####################### #
    from dataset.data_loader import get_dataloaders
    dataset = get_dataloaders(cfg)
    train_loader = dataset['train']
    if cfg.evaluate:
        val_loader = dataset['valid']


    classifier_AU10=classifier_AU()
    classifier_AU14=classifier_AU()
    classifier_AU20=classifier_AU()
    classifier_AU25=classifier_AU()
    classifier_AU26=classifier_AU()
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
        
    optimizer_classifier = torch.optim.Adam(params=list(classifier_AU10.parameters())+list(classifier_AU14.parameters())+list(classifier_AU20.parameters())+list(classifier_AU25.parameters())+list(classifier_AU26.parameters()), lr=0.00000000001, betas=(0.5, 0.999))


    # ####################### Train ############################# #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        rec_loss_train, quant_loss_train, pp_train = train(train_loader, classifier_AU10,classifier_AU14,classifier_AU20,classifier_AU25,classifier_AU26, calc_vq_loss,optimizer_classifier, epoch, cfg)
        epoch_log = epoch + 1
        if main_process(cfg):
            logger.info('TRAIN Epoch: {} '
                        'loss_train: {} '
                        'pp_train: {} '
                        .format(epoch_log, rec_loss_train, pp_train)
                        )
            for m, s in zip([rec_loss_train, quant_loss_train, pp_train],
                            ["train/rec_loss", "train/quant_loss", "train/perplexity"]):
                writer.add_scalar(s, m, epoch_log)





        if (epoch_log % cfg.save_freq == 0) and main_process(cfg):
            save_checkpoint(classifier_AU10,
                            sav_path=os.path.join(cfg.save_path, 'classifier_model10')
                            )
            save_checkpoint(classifier_AU14,
                            sav_path=os.path.join(cfg.save_path, 'classifier_model14')
                            )
            save_checkpoint(classifier_AU20,
                            sav_path=os.path.join(cfg.save_path, 'classifier_model20')
                            )
            save_checkpoint(classifier_AU25,
                            sav_path=os.path.join(cfg.save_path, 'classifier_model25')
                            )
            save_checkpoint(classifier_AU26,
                            sav_path=os.path.join(cfg.save_path, 'classifier_model26')
                            )

import numpy as np
import loss
def train(train_loader, classifier_AU10,classifier_AU14,classifier_AU20,classifier_AU25,classifier_AU26, loss_fn, optimizer, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    rec_loss_meter = AverageMeter()
    quant_loss_meter = AverageMeter()
    pp_meter = AverageMeter()

    end = time.time()
    max_iter = cfg.epochs * len(train_loader)
    for i, (data,mask,aus) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)
        data = data.cuda(cfg.gpu, non_blocking=True)
        #template = template.cuda(cfg.gpu, non_blocking=True)
        
        aus = torch.as_tensor(aus).cuda()
        
        gt_au10 = aus[:,0].unsqueeze(0)
        
        gt_au14 = aus[:,1].unsqueeze(0)
        gt_au20 = aus[:,2].unsqueeze(0)
        gt_au25 = aus[:,3].unsqueeze(0)
        gt_au26 = aus[:,4].unsqueeze(0)
        
        data=data.squeeze(0)
        classifier_AU10=classifier_AU10.cuda()
        classifier_AU10=classifier_AU14.cuda()
        classifier_AU10=classifier_AU20.cuda()
        classifier_AU10=classifier_AU25.cuda()
        classifier_AU10=classifier_AU26.cuda()
        linear2_au10,predict_au10=classifier_AU10(data)
        linear2_au14,predict_au14=classifier_AU14(data)
        linear2_au20,predict_au20=classifier_AU20(data)
        linear2_au25,predict_au25=classifier_AU25(data)
        linear2_au26,predict_au26=classifier_AU26(data)
        #print(gt_au10.shape,predict_au10.shape)
        #print(predict_au10.shape,gt_au10.shape,gt_au10[1].unsqueeze(0).shape)
        loss_au_dice10 =loss.au_dice_loss(predict_au10, gt_au10)
        loss_au_dice14 =loss.au_dice_loss(predict_au14, gt_au14)
        loss_au_dice20 =loss.au_dice_loss(predict_au20, gt_au20)
        loss_au_dice25 =loss.au_dice_loss(predict_au25, gt_au25)
        loss_au_dice26 =loss.au_dice_loss(predict_au26, gt_au26)

        '''
        predict_au10 = predict_au10.contiguous().view(predict_au10.shape[0])
        predict_au14 = predict_au14.contiguous().view(predict_au14.shape[0])
        predict_au20 = predict_au20.contiguous().view(predict_au20.shape[0])
        predict_au25 = predict_au25.contiguous().view(predict_au25.shape[0])
        predict_au26 = predict_au26.contiguous().view(predict_au26.shape[0])
        '''
        AU_loss = nn.BCELoss()
        #print(predict_au10.shape)
        #print(gt_au10.shape,predict_au10.shape)
        classifierloss_au10 = AU_loss(predict_au10.float(),gt_au10.float())
        classifierloss_au14 = AU_loss(predict_au14.float(),gt_au14.float())
        classifierloss_au20 = AU_loss(predict_au20.float(),gt_au20.float())
        classifierloss_au25 = AU_loss(predict_au25.float(),gt_au25.float())
        classifierloss_au26 = AU_loss(predict_au26.float(),gt_au26.float())
        
        loss_au10= (loss_au_dice10+ classifierloss_au10)
        loss_au14= (loss_au_dice14+ classifierloss_au14)
        loss_au20= (loss_au_dice20+ classifierloss_au20)
        loss_au25= (loss_au_dice25+ classifierloss_au25)
        loss_au26= (loss_au_dice26+ classifierloss_au26)



        whole_AU_loss=(loss_au10+loss_au14+loss_au20+loss_au25+loss_au26)/5
        
        optimizer.zero_grad()
        whole_AU_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        
        # Adjust lr


        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % cfg.print_freq == 0 and main_process(cfg):
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain: {remain_time} '
                        'Loss: {loss_meter:.4f} '
                        .format(epoch + 1, cfg.epochs, i + 1, len(train_loader),
                                batch_time=batch_time, data_time=data_time,
                                remain_time=remain_time,
                                loss_meter=whole_AU_loss.item()
                                ))

    return rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg


def validate(val_loader, model, loss_fn, epoch, cfg):
    rec_loss_meter = AverageMeter()
    quant_loss_meter = AverageMeter()
    pp_meter = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (data,mask) in enumerate(val_loader):
            data = data.cuda(cfg.gpu, non_blocking=True)
            #template = template.cuda(cfg.gpu, non_blocking=True)

            data=data.squeeze(0)
            out, quant_loss, info = model(data)
            out.permute(1, 2, 0)[mask[0]<0.5]=1.0

            # LOSS
            loss, loss_details = loss_fn(out, data.squeeze(0), quant_loss, quant_loss_weight=cfg.quant_loss_weight)

            if cfg.distributed:
                loss = reduce_tensor(loss, cfg)


            for m, x in zip([rec_loss_meter, quant_loss_meter, pp_meter],
                            [loss_details[0], loss_details[1], info[0]]):
                m.update(x.item(), 1) #batch_size = 1 for validation


    return rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg


if __name__ == '__main__':
    main()
