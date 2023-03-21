import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import argparse
import cv2
import time

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from models.vgg import vgg16
from utils.my_optim import reduce_lr
from utils.avgMeter import AverageMeter
from utils.LoadData import train_data_loader, valid_data_loader
from utils.Metrics import Cls_Accuracy, IOUMetric
from utils.util import output_visualize
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
from medpy import metric

def get_arguments():
    parser = argparse.ArgumentParser(description='classification pytorch implementation')
    parser.add_argument("--img_dir", type=str, default='', help='Directory of training images')
    parser.add_argument("--train_list", type=str, default='dataset/train.txt')
    parser.add_argument("--test_list", type=str, default='dataset/valid.txt')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=448)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--decay_points", type=str, default='50,100')
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument('--logdir', default='result/logs', type=str, help='Tensorboard log dir')
    parser.add_argument('--save_folder', default='result/checkpoints', help='Location to save checkpoint models')
    parser.add_argument('--show_interval', default=50, type=int, help='interval of showing training conditions')
    parser.add_argument("--global_counter", type=int, default=0)

    return parser.parse_args()


def get_model(args):
    model = vgg16(pretrained=True) 

    model = torch.nn.DataParallel(model).cuda()
    param_groups = model.module.parameters()
    
    optimizer = optim.SGD([
        {'params': param_groups, 'lr': args.lr}], 
        momentum=0.9, 
        weight_decay=args.weight_decay, 
        nesterov=True
    )

    return  model, optimizer


def validate(current_epoch):
    print('\nvalidating ... ', flush=True, end='')
    
    cls_acc_matrix = Cls_Accuracy()
    
    val_loss = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
            img, label, _, _, _ = dat
            label = label.to('cuda', non_blocking=True)
            img = img.to('cuda', non_blocking=True)
            
            logit = model(img)

            """ classification loss """
            loss = F.multilabel_soft_margin_loss(logit, label)
            cls_acc_matrix.update(logit, label)

            val_loss.update(loss.data.item(), img.size()[0])            
            
    """ validation score """
    val_cls_acc = cls_acc_matrix.compute_avg_acc()
        
    writer.add_scalar('valid loss', val_loss.avg, current_epoch)
    writer.add_scalar('valid acc', val_cls_acc, current_epoch)
    
    print('validating loss: %.4f' % val_loss.avg)
    print('validating acc: %.4f' % val_cls_acc)
    
    return val_cls_acc
    

def train(current_epoch):
    train_loss = AverageMeter()
    cls_acc_matrix = Cls_Accuracy()

    model.train()
    
    global_counter = args.global_counter

    """ learning rate decay """
    res = reduce_lr(args, optimizer, current_epoch)

    for idx, dat in enumerate(train_loader):

        img, label, _ = dat   

        label = label.to('cuda', non_blocking=True)
        img = img.to('cuda', non_blocking=True)

        logit = model(img)

        """ classification loss """
        loss = F.multilabel_soft_margin_loss(logit, label)

        """ backprop """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cls_acc_matrix.update(logit, label)
        train_loss.update(loss.data.item(), img.size()[0])
        
        global_counter += 1

        """ tensorboard log """
        if global_counter % args.show_interval == 0:
            train_cls_acc = cls_acc_matrix.compute_avg_acc()

            writer.add_scalar('train loss', train_loss.avg, global_counter)
            writer.add_scalar('train acc', train_cls_acc, global_counter)

            print('Epoch: [{}][{}/{}]\t'
                  'LR: {:.5f}\t'
                  'ACC: {:.5f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    current_epoch, idx+1, len(train_loader),
                    optimizer.param_groups[0]['lr'], 
                    train_cls_acc, loss=train_loss)
                 )

    args.global_counter = global_counter
    
    
if __name__ == '__main__':
    args = get_arguments()
    nGPU = torch.cuda.device_count()
    print("start training the classifier with DRS , nGPU = %d" % nGPU)
    
    args.batch_size *= nGPU
    args.num_workers *= nGPU
    
    print('Running parameters:\n', args)
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    writer = SummaryWriter(log_dir=args.logdir)
    
    train_loader = train_data_loader(args)
    val_loader = valid_data_loader(args)
    print('# of train dataset:', len(train_loader) * args.batch_size)
    # print('# of valid dataset:', len(val_loader) * args.batch_size)
    print('# of valid dataset:', len(val_loader) * 1)
    print()

    best_score = 0
    model, optimizer = get_model(args)
    print(model)
    score = validate(0)
    for current_epoch in range(1, args.epoch+1):
        
        train(current_epoch)
        score = validate(current_epoch)
        
        """ save checkpoint """
        if score > best_score:
            best_score = score
            print('\nSaving state, epoch : %d , acc : %.4f \n' % (current_epoch, score))
            state = {
                'model': model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                'epoch': current_epoch,
                'iter': args.global_counter,
                'miou': score,
            }
            model_file = os.path.join(args.save_folder, 'best.pth')
            torch.save(state, model_file)
