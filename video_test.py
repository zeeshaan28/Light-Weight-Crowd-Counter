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
import cv2




def detect(checkpoint=' ',
           source=' ',
           transform=True,
           gpu=' '):

    seed = time.time()
    
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
            
            

        test(source, model)


def test(video_file, model):
    
    print('begin test')
    
    
    mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)])
    
    model.eval()
    
    cap = cv2.VideoCapture(video_file)
    i = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == False:
            break
        im0 = img.copy()
        img = img_transform(img)[None, :, :, :]
        
        with torch.no_grad():
            img = Variable(img).cuda()
            output = model(img).detach()
            output = output.squeeze(0).squeeze(0).cpu().numpy()
            count = round(np.sum(output))
            
        crowd_count = 'Crowd Count: ' + str(count)
        
            
        cv2.putText(im0,crowd_count,(10,30),cv2.FONT_HERSHEY_PLAIN, 2, [0, 255, 0], 2)   
        cv2.imshow(str(video_file), im0)
        cv2.waitKey(1) 
        
            

    cap.release()
    cv2.destroyAllWindows()
    
        
                 
def parse_opt():
    
    parser = argparse.ArgumentParser(description='PyTorch Crowd Counter')
    
    parser.add_argument('--checkpoint', default='model.pth.tar', type=str,
                        help='path to the checkpoint')
    
    parser.add_argument('--source', type=str,
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