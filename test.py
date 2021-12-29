import sys
import os

import warnings

from models.model_vgg import CSRNet as CSRNet_vgg
from models.model_student_vgg import CSRNet as CSRNet_student

from utils import save_checkpoint
from utils import cal_para, crop_img_patches

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.transforms as standard_transforms
import json
from PIL import Image, ImageOps
from glob import glob

import numpy as np
import argparse
import json
import dataset
import time




def detect(checkpoint=' ',
           data_path=' ',
           transform=True,
           gpu=' '):

    seed = time.time()
    
    files_list = glob(data_path + '/*')
    
    if transform == 'false':
        transform = False

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.manual_seed(seed)

    model = CSRNet_student(ratio=4, transform=transform)
    print (model)
    #cal_para(model)  # including 1x1conv transform layer that can be removed
   
    model = model.cuda()

    if checkpoint:
        
        if os.path.isfile(checkpoint):
            print("=> loading checkpoint '{}'".format(checkpoint))
            checkpoint = torch.load(checkpoint)

            if transform is False:
                # remove 1x1 conv para
                for k in checkpoint['state_dict'].keys():
                    if k[:9] == 'transform':
                        del checkpoint['state_dict'][k]

            model.load_state_dict(checkpoint['state_dict'])
            #print("=> loaded checkpoint '{}' (epoch {})"
                  #.format(checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint))
            
            

        test(files_list, model)


def test(test_list, model):
    
    print('begin test')
    
    
    mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)])
    
    model.eval()
    
    for file in test_list:
    
        img = Image.open(file)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)[None, :, :, :]
        with torch.no_grad():
            img = Variable(img).cuda()
            output = model(img)
            pred = output.data.sum()

        print(f'{file} {pred:.2f}')

       
def parse_opt():
    
    parser = argparse.ArgumentParser(description='PyTorch Crowd Counter')
    
    parser.add_argument('--checkpoint', default='model.pth.tar', type=str,
                        help='path to the checkpoint')
    
    parser.add_argument('--data_path', default='testing', type=str,
                        help='path to the inference')
    
    parser.add_argument('--transform', default=True, type=str,
                        help='1x1 conv transform')
    
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')
    
    opt = parser.parse_args()
    
    return opt

def main(opt):
    detect(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)