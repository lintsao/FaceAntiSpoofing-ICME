# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from PIL import Image
import sys
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Function
import argparse
import csv
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
import torch.optim as optim
import time
import random
import math
from scipy.optimize import brentq
from scipy.interpolate import interp1d

"""## Dataset"""

#    OULU          Users	 Real access	 Print attacks	 Video attacks	 Total
#  Training	        20	      360           720	            720           1800 有少，只有 1253 (只偵測到 1792 人臉)
#  Development	    15	      270	          540	            540	          1350
#  Test	            20	      360	          720	            720	          1800 有少，只有 1799 v
################################################################################
#    CASIA         Users	 Real access	 Print attacks	 Video attacks	 Total
#  Training	        20	      60           	 120              60           240 有少，只有 239 v
#  Test	            30	      90	           180	            90	         360 v
################################################################################
#    MSU         Users	 Real access	 Print attacks	 Video attacks	 Total
#  Training	        15	      30           	 30              60          120 v
#  Test	            20	      40	           40	             80	         160 v
################################################################################
#    IDIAP         Users	 Real access	 Print attacks	 Video attacks	 Total
#  Training	        ?	      60           	 60             240           360 目前用 fixed: 60/ 30/ 120
#  Development	    ?	      60	           60	            240	          360
#  Test	            ?	      80	           80	            320	          480 有少，只有 479 v
################################################################################

class Oulu_dataset(Dataset): 
    def __init__(self, root, mode, transform = None, transform_depth = None, attack = None):
        self.common_path = root
        self.mode = mode # train, eval, test
        self.transform = transform
        self.transform_depth = transform_depth
        self.attack = attack
        self.filename = sorted(os.listdir(self.common_path))
        self.pathname = [number for number in self.filename if 'jpg' in number and 'depth' not in number and '1_3_11_5' not in number]
        if self.mode == 'train':
            if self.attack == "print":
                self.pathname = [name for name in self.pathname if name.split("_")[4][0] == '2' or name.split("_")[4][0] == '3']
            elif self.attack =="real":
                self.pathname = [name for name in self.pathname if name.split("_")[4][0] == '1']
            elif self.attack == "replay":
                self.pathname = [name for name in self.pathname if name.split("_")[4][0] == '4' or name.split("_")[4][0] == '5']
        else:
            self.pathname = self.pathname  
        self.len = len(self.pathname) #全部pic個數
        print('OULU {} {}: {}'.format(self.mode, self.attack, self.len))

    def __getitem__(self, index):
        pic = str(self.pathname[index])[:-4] #cropped_1_1_01_1
        depth = pic + '_depth.jpg' #cropped_1_1_01_1_depth.jpg
        pic = pic + '.jpg' #cropped_1_1_01_1.jpg
        depth_path = os.path.join(self.common_path, depth)
        pic_path = os.path.join(self.common_path, pic)

        if self.mode == 'train':
            if self.pathname[index].split("_")[4][0] == '1': #real pic
                label = int(0)
            elif self.pathname[index].split("_")[4][0] == '2' or self.pathname[index].split("_")[4][0] == '3': #print12
                label = int(1)
            elif self.pathname[index].split("_")[4][0] == '4' or self.pathname[index].split("_")[4][0] == '5': #replay12
                label = int(2)
        else: #test,dev
            if self.pathname[index].split("_")[4][0] == '1': #real pic
                label = int(0)
            elif self.pathname[index].split("_")[4][0] == '2' or self.pathname[index].split("_")[4][0] == '3': #print12
                label = int(1)
            elif self.pathname[index].split("_")[4][0] == '4' or self.pathname[index].split("_")[4][0] == '5': #replay12
                label = int(2)
            im = Image.open(pic_path)                 
            if self.transform is not None:
                im = self.transform(im)
            return im, torch.tensor(label, dtype=torch.long)

        im = Image.open(pic_path)                 
        if self.transform is not None:
            im = self.transform(im)
        depth_im = Image.open(depth_path)
        if self.transform_depth is not None:
            depth_im = self.transform_depth(depth_im)
        return im, depth_im, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return(self.len)

class MSU_dataset(Dataset): 
    def __init__(self, root, mode, transform = None, transform_depth = None, attack = None):
        self.common_path = root   
        self.mode = mode # 在msu沒用 train, eval, test
        self.transform = transform
        self.transform_depth = transform_depth
        self.attack = attack
        self.train_list = ['002', '003', '005', '006', '007', '008', '009', '011', '012', '021', '022', '034', '053', '054', '055'] # train id
        self.filename = sorted(os.listdir(self.common_path))
        self.pathname = [number for number in self.filename if 'jpg' in number and 'depth' not in number]
        pathname = [number for number in self.filename if 'jpg' in number and 'depth' not in number]
        if self.mode == 'train':
            for file in pathname:
                if file.split('_')[2][-3:] not in self.train_list:
                    self.pathname.remove(file)
            if self.attack == "print":
                self.pathname = [name for name in self.pathname if "print" in name]
            elif self.attack =="real":
                self.pathname = self.pathname
            elif self.attack == "replay":
                self.pathname = [name for name in self.pathname if "print" not in name]
        else:
            for file in pathname:
                if file.split('_')[2][-3:] in self.train_list:
                    self.pathname.remove(file)
            if self.attack == "print":
                self.pathname = [name for name in self.pathname if "print" in name]
            elif self.attack =="real":
                self.pathname = self.pathname
            elif self.attack == "replay":
                self.pathname = [name for name in self.pathname if "print" not in name]            
        self.len = len(self.pathname) #全部pic個數
        print('MSU {} {}: {}'.format(self.mode, self.attack, self.len))

    def __getitem__(self, index):
        pic = str(self.pathname[index])[:-4] #cropped_attack_client001_android_SD_ipad_video_scene01
        depth = pic + '_depth.jpg' #cropped_attack_client001_android_SD_ipad_video_scene01_depth.jpg
        pic = pic + '.jpg' #cropped_attack_client001_android_SD_ipad_video_scene01.jpg
        depth_path = os.path.join(self.common_path, depth)
        pic_path = os.path.join(self.common_path, pic)

        if self.mode == 'train':
            if 'real' in self.common_path:
                    label = int(0)
            else: #attack
                if 'print' in pic: #print
                    label = int(1)
                else: #replay
                    label = int(2)   
        else:
            if 'real' in self.common_path:
                    label = int(0)
            else: #attack
                if 'print' in pic: #print
                    label = int(1)
                else: #replay
                    label = int(2) 
            im = Image.open(pic_path)                 
            if self.transform is not None:
                im = self.transform(im)
            return im, torch.tensor(label, dtype=torch.long)              
        
        im = Image.open(pic_path)                 
        if self.transform is not None:
            im = self.transform(im)
        depth_im = Image.open(depth_path)
        if self.transform_depth is not None:
            depth_im = self.transform_depth(depth_im)
        return im, depth_im, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return(self.len)

class Idiap_dataset(Dataset): 
    def __init__(self, root, mode, transform = None, transform_depth = None, attack= None):
        self.common_path = root   
        self.mode = mode 
        self.transform = transform
        self.transform_depth = transform_depth
        self.attack = attack
        self.filename = sorted(os.listdir(self.common_path))
        self.pathname = [number for number in self.filename if 'jpg' in number and 'depth' not in number]
        if self.mode == 'train':
            if self.attack == "print":
              self.pathname = [name for name in self.pathname if "print" in name]
            elif self.attack =="replay":
              self.pathname = [name for name in self.pathname if "print" not in name]
        else:
            if self.attack == "print":
              self.pathname = [name for name in self.pathname if "print" in name]
            elif self.attack =="replay":
              self.pathname = [name for name in self.pathname if "print" not in name]    
        self.len = len(self.pathname) #全部pic個數
        print('IDIAP {} {}: {}'.format(self.mode, self.attack, self.len))

    def __getitem__(self, index):
        pic = str(self.pathname[index])[:-4] #ccropped_attack_highdef_client009_session01_highdef_photo_adverse
        depth = pic + '_depth.jpg' #cropped_attack_highdef_client009_session01_highdef_photo_adverse_depth.jpg
        pic = pic + '.jpg' #ccropped_attack_highdef_client009_session01_highdef_photo_adverse.jpg
        depth_path = os.path.join(self.common_path, depth)
        pic_path = os.path.join(self.common_path, pic)

        if self.mode == 'train':
            if 'real' in self.common_path:
                    label = int(0)
            else: #attack
                if 'print' in pic: #print
                    label = int(1)
                else: #replay
                    label = int(2)      
        else:
            if 'real' in self.common_path:
                    label = int(0)
            else: #attack
                if 'print' in pic: #print
                    label = int(1)
                else: #replay
                    label = int(2)
            im = Image.open(pic_path)                
            if self.transform is not None:
                im = self.transform(im)
            return im, torch.tensor(label, dtype=torch.long)          
            
        im = Image.open(pic_path)                
        if self.transform is not None:
            im = self.transform(im)
        depth_im = Image.open(depth_path)
        if self.transform_depth is not None:
            depth_im = self.transform_depth(depth_im)
        return im, depth_im, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return(self.len)

class Casia_dataset(Dataset): 
    def __init__(self, root, mode, transform = None, transform_depth = None, attack = None):
        self.common_path = root   
        self.mode = mode # train, eval, test
        self.transform = transform
        self.transform_depth = transform_depth
        self.attack = attack
        self.filename = sorted(os.listdir(self.common_path))
        self.pathname = [number for number in self.filename if 'jpg' in number and 'depth' not in number]
        if self.mode == 'train':
            if self.attack == "print":
                self.pathname = [name for name in self.pathname if (('HR' in name and (name.split("_")[3][0] == '2' or name.split("_")[3][0] == '3')) or 
                                 ('HR' not in name and (name.split("_")[2][0] == '3' or name.split("_")[2][0] == '4' or name.split("_")[2][0] == '5' or name.split("_")[2][0] == '6')))]
            elif self.attack =="real":
                self.pathname = [name for name in self.pathname if (('HR' in name and (name.split("_")[3][0] == '1')) or 
                                 ('HR' not in name and (name.split("_")[2][0] == '1' or name.split("_")[2][0] == '2')))]
            elif self.attack == "replay":
                self.pathname = [name for name in self.pathname if (('HR' in name and (name.split("_")[3][0] == '4')) or 
                                 ('HR' not in name and (name.split("_")[2][0] == '7' or name.split("_")[2][0] == '8')))]
        else:
            self.pathname = self.pathname      
        self.len = len(self.pathname) #全部pic個數
        print('CASIA {} {}: {}'.format(self.mode, self.attack, self.len))

    def __getitem__(self, index):
        pic = str(self.pathname[index])[:-4] #cropped_attack_client001_android_SD_ipad_video_scene01
        depth = pic + '_depth.jpg' #cropped_attack_client001_android_SD_ipad_video_scene01_depth.jpg
        pic = pic + '.jpg' #cropped_attack_client001_android_SD_ipad_video_scene01.jpg
        depth_path = os.path.join(self.common_path, depth)
        pic_path = os.path.join(self.common_path, pic)

        #train
        if self.mode == 'train':
            if 'HR' in self.pathname[index]:
                if self.pathname[index].split("_")[3][0] == '1': #real pic
                    label = int(0)
                elif self.pathname[index].split("_")[3][0] == '2' or self.pathname[index].split("_")[3][0] == '3': #print
                    label = int(1)
                elif self.pathname[index].split("_")[3][0] == '4': #replay
                    label = int(2)
            else:
                if self.pathname[index].split("_")[2][0] == '1' or self.pathname[index].split("_")[2][0] == '2': #real pic
                    label = int(0)
                elif self.pathname[index].split("_")[2][0] == '3' or self.pathname[index].split("_")[2][0] == '4' or self.pathname[index].split("_")[2][0] == '5' or self.pathname[index].split("_")[2][0] == '6': #print
                    label = int(1)
                elif self.pathname[index].split("_")[2][0] == '7' or self.pathname[index].split("_")[2][0] == '8': #replay
                    label = int(2)
        else: #test,dev
            if 'HR' in self.pathname[index]: # use acc
                if self.pathname[index].split("_")[3][0] == '1': #real pic
                    label = int(0)
                elif self.pathname[index].split("_")[3][0] == '2' or self.pathname[index].split("_")[3][0] == '3': #print
                    label = int(1)
                elif self.pathname[index].split("_")[3][0] == '4': #replay
                    label = int(2)
            else:
                if self.pathname[index].split("_")[2][0] == '1' or self.pathname[index].split("_")[2][0] == '2': #real pic
                    label = int(0)
                elif self.pathname[index].split("_")[2][0] == '3' or self.pathname[index].split("_")[2][0] == '4' or self.pathname[index].split("_")[2][0] == '5' or self.pathname[index].split("_")[2][0] == '6': #print
                    label = int(1)
                elif self.pathname[index].split("_")[2][0] == '7' or self.pathname[index].split("_")[2][0] == '8': #replay
                    label = int(2)
            im = Image.open(pic_path)                
            if self.transform is not None:
                im = self.transform(im)
            return im, torch.tensor(label, dtype=torch.long)

        im = Image.open(pic_path)                
        if self.transform is not None:
            im = self.transform(im)
        depth_im = Image.open(depth_path)
        if self.transform_depth is not None:
            depth_im = self.transform_depth(depth_im) 
        return im, depth_im, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return(self.len)

def choose_dataset(path, target_domain, img_size, depth_size):

    transform = transforms.Compose([
        transforms.Resize((img_size ,img_size)), # 隨機將圖片水平翻轉
        transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
        transforms.Normalize([0.0,],[1.0,])
    ])

    transform_depth = transforms.Compose([
        transforms.Resize((depth_size ,depth_size)), # 隨機將圖片水平翻轉
        transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
        transforms.Normalize([0.0,],[1.0,])
    ])

    ### test dataset ###
    msu_train_real_dataset = MSU_dataset(os.path.join(path, 'pr_depth_map/MSU/real/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "real")
    msu_train_print_dataset = MSU_dataset(os.path.join(path, 'pr_depth_map/MSU/attack/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "print")
    msu_train_replay_dataset = MSU_dataset(os.path.join(path, 'pr_depth_map/MSU/attack/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "replay")
    msu_test_real_dataset = MSU_dataset(os.path.join(path, 'MSU/dataset/scene01/real/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "real")
    msu_test_print_dataset = MSU_dataset(os.path.join(path, 'MSU/dataset/scene01/attack/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "print")
    msu_test_replay_dataset = MSU_dataset(os.path.join(path, 'MSU/dataset/scene01/attack/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "replay")

    idiap_train_real_dataset = Idiap_dataset(os.path.join(path, 'pr_depth_map/ReplayAttack/replayattack-train/real/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "real")
    idiap_train_print_fixed_dataset = Idiap_dataset(os.path.join(path, 'pr_depth_map/ReplayAttack/replayattack-train/attack/fixed/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "print")
    idiap_train_replay_fixed_dataset = Idiap_dataset(os.path.join(path, 'pr_depth_map/ReplayAttack/replayattack-train/attack/fixed/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "replay")
    idiap_train_print_hand_dataset = Idiap_dataset(os.path.join(path, 'pr_depth_map/ReplayAttack/replayattack-train/attack/hand/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "print")
    idiap_train_replay_hand_dataset = Idiap_dataset(os.path.join(path, 'pr_depth_map/ReplayAttack/replayattack-train/attack/hand/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "replay")
    idiap_test_real_dataset = Idiap_dataset(os.path.join(path, 'Replay_Attack/replayattack-test/test/real/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "real")
    idiap_test_print_fixed_dataset = Idiap_dataset(os.path.join(path, 'Replay_Attack/replayattack-test/test/attack/fixed/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "print")
    idiap_test_replay_fixed_dataset = Idiap_dataset(os.path.join(path, 'Replay_Attack/replayattack-test/test/attack/fixed/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "replay")
    idiap_test_print_hand_dataset = Idiap_dataset(os.path.join(path, 'Replay_Attack/replayattack-test/test/attack/hand/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "print")
    idiap_test_replay_hand_dataset = Idiap_dataset(os.path.join(path, 'Replay_Attack/replayattack-test/test/attack/hand/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "replay")

    oulu_train_real_dataset = Oulu_dataset(os.path.join(path, 'pr_depth_map/Oulu_NPU/Train_files/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack='real')
    oulu_train_print_dataset = Oulu_dataset(os.path.join(path, 'pr_depth_map/Oulu_NPU/Train_files/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "print")
    oulu_train_replay_dataset = Oulu_dataset(os.path.join(path, 'pr_depth_map/Oulu_NPU/Train_files/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "replay")
    oulu_test_dataset = Oulu_dataset(os.path.join(path, 'Oulu_NPU/Test_files/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth)

    casia_train_real_dataset = Casia_dataset(os.path.join(path, 'pr_depth_map/CASIA_faceAntisp/train_release/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack='real')
    casia_train_print_dataset = Casia_dataset(os.path.join(path, 'pr_depth_map/CASIA_faceAntisp/train_release/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack='print')
    casia_train_replay_dataset = Casia_dataset(os.path.join(path, 'pr_depth_map/CASIA_faceAntisp/train_release/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack='replay')
    casia_test_dataset = Casia_dataset(os.path.join(path, 'CASIA_faceAntisp/test_release/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth)


    if target_domain == 'msu': 
        test_dataset = msu_test_real_dataset + msu_test_print_dataset + msu_test_replay_dataset
        domain1_real_dataset = idiap_train_real_dataset
        domain1_print_dataset = idiap_train_print_fixed_dataset + idiap_train_print_hand_dataset
        domain1_replay_dataset = idiap_train_replay_fixed_dataset + idiap_train_replay_hand_dataset
        domain2_real_dataset = oulu_train_real_dataset
        domain2_print_dataset = oulu_train_print_dataset
        domain2_replay_dataset = oulu_train_replay_dataset
        domain3_real_dataset = casia_train_real_dataset
        domain3_print_dataset = casia_train_print_dataset
        domain3_replay_dataset = casia_train_replay_dataset
    elif target_domain == 'idiap':
        test_dataset = idiap_test_real_dataset + idiap_test_print_fixed_dataset + idiap_test_replay_fixed_dataset + idiap_test_print_hand_dataset + idiap_test_replay_hand_dataset 
        domain1_real_dataset = msu_train_real_dataset
        domain1_print_dataset = msu_train_print_dataset
        domain1_replay_dataset = msu_train_replay_dataset
        domain2_real_dataset = oulu_train_real_dataset
        domain2_print_dataset = oulu_train_print_dataset
        domain2_replay_dataset = oulu_train_replay_dataset
        domain3_real_dataset = casia_train_real_dataset
        domain3_print_dataset = casia_train_print_dataset
        domain3_replay_dataset = casia_train_replay_dataset     
    elif target_domain == 'oulu':
        test_dataset = oulu_test_dataset
        domain1_real_dataset = msu_train_real_dataset
        domain1_print_dataset = msu_train_print_dataset
        domain1_replay_dataset = msu_train_replay_dataset
        domain2_real_dataset = idiap_train_real_dataset
        domain2_print_dataset = idiap_train_print_fixed_dataset + idiap_train_print_hand_dataset
        domain2_replay_dataset = idiap_train_replay_fixed_dataset + idiap_train_replay_hand_dataset
        domain3_real_dataset = casia_train_real_dataset
        domain3_print_dataset = casia_train_print_dataset
        domain3_replay_dataset = casia_train_replay_dataset   
    elif target_domain == 'casia':
        test_dataset = casia_test_dataset
        domain1_real_dataset = msu_train_real_dataset
        domain1_print_dataset = msu_train_print_dataset
        domain1_replay_dataset = msu_train_replay_dataset
        domain2_real_dataset = idiap_train_real_dataset
        domain2_print_dataset = idiap_train_print_fixed_dataset + idiap_train_print_hand_dataset
        domain2_replay_dataset = idiap_train_replay_fixed_dataset + idiap_train_replay_hand_dataset
        domain3_real_dataset = oulu_train_real_dataset
        domain3_print_dataset = oulu_train_print_dataset
        domain3_replay_dataset = oulu_train_replay_dataset
    return  test_dataset, domain1_real_dataset, domain1_print_dataset, domain1_replay_dataset, \
            domain2_real_dataset, domain2_print_dataset, domain2_replay_dataset, domain3_real_dataset, \
            domain3_print_dataset, domain3_replay_dataset