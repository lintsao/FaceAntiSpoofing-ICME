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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
path = 'drive/My Drive/senior_2/Face_AntiSpoofing'
number_folder = '12'
target_domain = 'oulu'
 
# target:MSU    => 1: i, 2: o, 3: c
# target:Idiap  => 1: m, 2: o, 3: c
# target:Oulu   => 1: m, 2: i, 3: c
# target:Casia  => 1: m, 2: i, 3: o
def make_model_path(number_folder, target_domain):
    if not os.path.exists(os.path.join(path, 'model', number_folder)):
        os.makedirs(os.path.join(path, 'model', number_folder))

    if not os.path.exists(os.path.join(path, 'model', number_folder, target_domain)):
        os.makedirs(os.path.join(path, 'model', number_folder, target_domain))

    shared_spoof_path = os.path.join(path, 'model', number_folder, target_domain, '{}_spoof_encoder.pt'.format(target_domain))
    spoof_classify_path = os.path.join(path, 'model', number_folder, target_domain, '{}_spoof_classify.pt'.format(target_domain))
    shared_content_path = os.path.join(path, 'model', number_folder, target_domain, '{}_content_encoder.pt'.format(target_domain))
    if target_domain == 'msu':           
        domain1_encoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_i_domain_encoder.pt'.format(target_domain))
        domain2_encoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_o_domain_encoder.pt'.format(target_domain))
        domain3_encoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_c_domain_encoder.pt'.format(target_domain))
    elif target_domain == 'idiap':
        domain1_encoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_m_domain_encoder.pt'.format(target_domain))
        domain2_encoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_o_domain_encoder.pt'.format(target_domain))
        domain3_encoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_c_domain_encoder.pt'.format(target_domain))
    elif target_domain == 'oulu':
        domain1_encoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_m_domain_encoder.pt'.format(target_domain))
        domain2_encoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_i_domain_encoder.pt'.format(target_domain))
        domain3_encoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_c_domain_encoder.pt'.format(target_domain))
    elif target_domain == 'casia':
        domain1_encoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_m_domain_encoder.pt'.format(target_domain))
        domain2_encoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_i_domain_encoder.pt'.format(target_domain))
        domain3_encoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_o_domain_encoder.pt'.format(target_domain))
    domain_classify_path = os.path.join(path, 'model', number_folder, target_domain, '{}_domain_classify.pt'.format(target_domain))
    decoder_path = os.path.join(path, 'model', number_folder, target_domain, '{}_decoder.pt'.format(target_domain))
    depth_map_path = os.path.join(path, 'model', number_folder, target_domain, '{}_depth_map.pt'.format(target_domain))
    return shared_spoof_path, spoof_classify_path, shared_content_path, domain1_encoder_path, domain2_encoder_path, domain3_encoder_path, \
    domain_classify_path, decoder_path, depth_map_path

shared_spoof_path, spoof_classify_path, shared_content_path, \
domain1_encoder_path, domain2_encoder_path, domain3_encoder_path, \
domain_classify_path, decoder_path, depth_map_path = make_model_path(number_folder, target_domain) 

print('目前測試 Model名稱 : ', number_folder, '目前測試 target 名稱 : ', target_domain)
random.seed(307)
np.random.seed(307)
torch.manual_seed(307)
torch.cuda.manual_seed(307)
torch.cuda.manual_seed_all(307)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""## Utils"""

def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def calculate(probs, labels):
    TN, FN, FP, TP = eval_state(probs, labels, 0.5)
    APCER = 1.0 if (FP + TN == 0) else FP / float(FP + TN)
    NPCER = 1.0 if (FN + TP == 0) else FN / float(FN + TP)
    ACER = (APCER + NPCER) / 2.0
    ACC = (TP + TN) / labels.shape[0]
    return APCER, NPCER, ACER, ACC

def calculate_threshold(probs, labels, threshold):
    TN, FN, FP, TP = eval_state(probs, labels, threshold)
    ACC = (TP + TN) / labels.shape[0]
    return ACC

def get_threshold(probs, grid_density):
    Min, Max = min(probs), max(probs)
    thresholds = []
    for i in range(grid_density + 1):
        thresholds.append(0.0 + i * 1.0 / float(grid_density))
    thresholds.append(1.1)
    return thresholds

def get_EER_states(probs, labels, grid_density = 10000):
    thresholds = get_threshold(probs, grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if(FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif(FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return EER, thr, FRR_list, FAR_list

def get_HTER_at_thr(probs, labels, thr):
    TN, FN, FP, TP = eval_state(probs, labels, thr)
    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN)
    elif(FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)
    HTER = (FAR + FRR) / 2.0
    return HTER

def HTER(y, y_pred):
    cur_EER_valid, threshold, _, _ = get_EER_states(y, y_pred)
    ACC_threshold = calculate_threshold(y, y_pred, threshold)
    cur_HTER_valid = get_HTER_at_thr(y, y_pred, threshold)
    return ACC_threshold, cur_HTER_valid

def sample_triplet(triplet_loss, d1_real, d1_print, d1_replay, d2_real, d2_print, d2_replay, d3_real, d3_print, d3_replay):
    output_loss = 0.0
    anchor_domain = list(np.random.choice(a=[1, 2, 3], size=6, replace=True, p=None))
    positive_domain = []
    negative_domain = []
    for index in range(len(anchor_domain)):
        domain_choice = [1, 2, 3]
        domain_choice.remove(anchor_domain[index])
        positive_domain.append(int(np.random.choice(a=domain_choice, size=1, replace=True, p=None)))
        negative_domain.append(int(np.random.choice(a=domain_choice, size=1, replace=True, p=None)))
        print(anchor_domain[index], positive_domain[index], negative_domain[index])
        if index == 0:    # rrp
            if anchor_domain[index] == 1:
                anchor = d1_real
            elif anchor_domain[index] == 2:
                anchor = d2_real
            else:
                anchor = d3_real
            if positive_domain[index] == 1:
                positive = d1_real
            elif positive_domain[index] == 2:
                positive = d2_real
            else:
                positive = d3_real
            if negative_domain[index] == 1:
                negative = d1_print
            elif negative_domain[index] == 2:
                negative = d2_print
            else:
                negative = d3_print          
        elif index == 1:  # ppR
            if anchor_domain[index] == 1:
                anchor = d1_print
            elif anchor_domain[index] == 2:
                anchor = d2_print
            else:
                anchor = d3_print
            if positive_domain[index] == 1:
                positive = d1_print
            elif positive_domain[index] == 2:
                positive = d2_print
            else:
                positive = d3_print
            if negative_domain[index] == 1:
                negative = d1_replay
            elif negative_domain[index] == 2:
                negative = d2_replay
            else:
                negative = d3_replay 
        elif index == 2:  # RRr
            if anchor_domain[index] == 1:
                anchor = d1_replay
            elif anchor_domain[index] == 2:
                anchor = d2_replay
            else:
                anchor = d3_replay
            if positive_domain[index] == 1:
                positive = d1_replay
            elif positive_domain[index] == 2:
                positive = d2_replay
            else:
                positive = d3_replay
            if negative_domain[index] == 1:
                negative = d1_real
            elif negative_domain[index] == 2:
                negative = d2_real
            else:
                negative = d3_real 
        elif index == 3:  # rrR
            if anchor_domain[index] == 1:
                anchor = d1_real
            elif anchor_domain[index] == 2:
                anchor = d2_real
            else:
                anchor = d3_real
            if positive_domain[index] == 1:
                positive = d1_real
            elif positive_domain[index] == 2:
                positive = d2_real
            else:
                positive = d3_real
            if negative_domain[index] == 1:
                negative = d1_replay
            elif negative_domain[index] == 2:
                negative = d2_replay
            else:
                negative = d3_replay
        elif index == 4:  # ppr
            if anchor_domain[index] == 1:
                anchor = d1_print
            elif anchor_domain[index] == 2:
                anchor = d2_print
            else:
                anchor = d3_print
            if positive_domain[index] == 1:
                positive = d1_print
            elif positive_domain[index] == 2:
                positive = d2_print
            else:
                positive = d3_print
            if negative_domain[index] == 1:
                negative = d1_real
            elif negative_domain[index] == 2:
                negative = d2_real
            else:
                negative = d3_real
        elif index == 5:  # RRp
            if anchor_domain[index] == 1:
                anchor = d1_replay
            elif anchor_domain[index] == 2:
                anchor = d2_replay
            else:
                anchor = d3_replay
            if positive_domain[index] == 1:
                positive = d1_replay
            elif positive_domain[index] == 2:
                positive = d2_replay
            else:
                positive = d3_replay
            if negative_domain[index] == 1:
                negative = d1_print
            elif negative_domain[index] == 2:
                negative = d2_print
            else:
                negative = d3_print
        # print((anchor.shape, positive.shape, negative.shape))
        print(triplet_loss(anchor, positive, negative))    
        output_loss += triplet_loss(anchor, positive, negative)
    return output_loss

"""## Dataset"""

#    OULU          Users	 Real access	 Print attacks	 Video attacks	 Total
#  Training	        20	      360           720	            720           1800 有少，只有 1253
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


img_size = 256
depth_size = 64
batch_size = 8
batch_triplet = 4

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
msu_train_real_dataset = MSU_dataset(os.path.join(path, 'data/pr_depth_map/MSU/real/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "real")
msu_train_print_dataset = MSU_dataset(os.path.join(path, 'data/pr_depth_map/MSU/attack/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "print")
msu_train_replay_dataset = MSU_dataset(os.path.join(path, 'data/pr_depth_map/MSU/attack/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "replay")
msu_test_real_dataset = MSU_dataset(os.path.join(path, 'data/MSU/MSU/real/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "real")
msu_test_print_dataset = MSU_dataset(os.path.join(path, 'data/MSU/MSU/attack/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "print")
msu_test_replay_dataset = MSU_dataset(os.path.join(path, 'data/MSU/MSU/attack/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "replay")

idiap_train_real_dataset = Idiap_dataset(os.path.join(path, 'data/pr_depth_map/ReplayAttack/replayattack-train/real/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "real")
idiap_train_print_dataset = Idiap_dataset(os.path.join(path, 'data/pr_depth_map/ReplayAttack/replayattack-train/attack/fixed/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "print")
idiap_train_replay_dataset = Idiap_dataset(os.path.join(path, 'data/pr_depth_map/ReplayAttack/replayattack-train/attack/fixed/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "replay")
idiap_test_real_dataset = Idiap_dataset(os.path.join(path, 'data/ReplayAttack/replayattack-test/real/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "real")
idiap_test_print_fixed_dataset = Idiap_dataset(os.path.join(path, 'data/ReplayAttack/replayattack-test/attack/fixed/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "print")
idiap_test_fixed_dataset = Idiap_dataset(os.path.join(path, 'data/ReplayAttack/replayattack-test/attack/fixed/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "replay")
idiap_test_print_hand_dataset = Idiap_dataset(os.path.join(path, 'data/ReplayAttack/replayattack-test/attack/hand/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "print")
idiap_test_hand_dataset = Idiap_dataset(os.path.join(path, 'data/ReplayAttack/replayattack-test/attack/hand/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "replay")

oulu_train_real_dataset = Oulu_dataset(os.path.join(path, 'data/pr_depth_map/Oulu_NPU/Train_files/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack='real')
oulu_train_print_dataset = Oulu_dataset(os.path.join(path, 'data/pr_depth_map/Oulu_NPU/Train_files/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "print")
oulu_train_replay_dataset = Oulu_dataset(os.path.join(path, 'data/pr_depth_map/Oulu_NPU/Train_files/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack = "replay")
oulu_test_dataset = Oulu_dataset(os.path.join(path, 'data/Oulu_NPU/Test_files/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth)

casia_train_real_dataset = Casia_dataset(os.path.join(path, 'data/pr_depth_map/CASIA_faceAntisp/train_release/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack='real')
casia_train_print_dataset = Casia_dataset(os.path.join(path, 'data/pr_depth_map/CASIA_faceAntisp/train_release/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack='print')
casia_train_replay_dataset = Casia_dataset(os.path.join(path, 'data/pr_depth_map/CASIA_faceAntisp/train_release/crop_frame/crop_face'), 'train', transform = transform, transform_depth = transform_depth, attack='replay')
casia_test_dataset = Casia_dataset(os.path.join(path, 'data/CASIA_faceAntisp/test_release/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth)

if target_domain == 'msu': 
    test_dataset = msu_test_real_dataset + msu_test_print_dataset + msu_test_replay_dataset
    domain1_real_dataset = idiap_train_real_dataset
    domain1_print_dataset = idiap_train_print_dataset
    domain1_replay_dataset = idiap_train_replay_dataset
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
    domain2_print_dataset = idiap_train_print_dataset
    domain2_replay_dataset = idiap_train_replay_dataset
    domain3_real_dataset = casia_train_real_dataset
    domain3_print_dataset = casia_train_print_dataset
    domain3_replay_dataset = casia_train_replay_dataset   
elif target_domain == 'casia':
    test_dataset = casia_test_dataset
    domain1_real_dataset = msu_train_real_dataset
    domain1_print_dataset = msu_train_print_dataset
    domain1_replay_dataset = msu_train_replay_dataset
    domain2_real_dataset = idiap_train_real_dataset
    domain2_print_dataset = idiap_train_print_dataset
    domain2_replay_dataset = idiap_train_replay_dataset
    domain3_real_dataset = oulu_train_real_dataset
    domain3_print_dataset = oulu_train_print_dataset
    domain3_replay_dataset = oulu_train_replay_dataset 

print("test_dataset:{}".format(len(test_dataset)))
print("domain1_dataset:{}".format(len(domain1_real_dataset + domain1_print_dataset + domain1_replay_dataset)))
print("domain2_dataset:{}".format(len(domain2_real_dataset + domain2_print_dataset + domain2_replay_dataset)))
print("domain3_dataset:{}".format(len(domain3_real_dataset + domain3_print_dataset + domain3_replay_dataset)))

test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False)
domain1_loader = DataLoader(domain1_real_dataset + domain1_print_dataset + domain1_replay_dataset, batch_size = batch_size, shuffle = True)
domain2_loader = DataLoader(domain2_real_dataset + domain2_print_dataset + domain2_replay_dataset, batch_size = batch_size, shuffle = True)
domain3_loader = DataLoader(domain3_real_dataset + domain3_print_dataset + domain3_replay_dataset, batch_size = batch_size, shuffle = True)

domain1_real_loader = DataLoader(domain1_real_dataset, batch_size = batch_triplet, shuffle = True)
domain1_print_loader = DataLoader(domain1_print_dataset, batch_size = batch_triplet, shuffle = True)
domain1_replay_loader = DataLoader(domain1_replay_dataset, batch_size = batch_triplet, shuffle = True)
domain2_real_loader = DataLoader(domain2_real_dataset, batch_size = batch_triplet, shuffle = True)
domain2_print_loader = DataLoader(domain2_print_dataset, batch_size = batch_triplet, shuffle = True)
domain2_replay_loader = DataLoader(domain2_replay_dataset, batch_size = batch_triplet, shuffle = True)
domain3_real_loader = DataLoader(domain3_real_dataset, batch_size = batch_triplet, shuffle = True)
domain3_print_loader = DataLoader(domain3_print_dataset, batch_size = batch_triplet, shuffle = True)
domain3_replay_loader = DataLoader(domain3_replay_dataset, batch_size = batch_triplet, shuffle = True)

"""## Loss"""

class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels, positive):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if positive == False:
            return wf # only return its features
            
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return wf, -torch.mean(L) # return features and loss

def lambda_function(gamma, epoch):
    return 2/(1 + math.exp(-gamma*epoch)) - 1

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

class SIMSE(nn.Module):
    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)
        return simse

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

"""### Triplet Loss"""

class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """
    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)

        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            # Get the hardest negative pairs
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            triplet_loss = torch.mean(triplet_loss)
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = _get_triplet_mask(labels).float()
            triplet_loss = loss * mask
  
            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss
        
def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    # got the dot product between all embeddings
    cor_mat = torch.matmul(x, x.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    norm_mat = cor_mat.diag() # return diag

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = F.relu(distances)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = torch.eq(distances, 0.0).float() # check whether distance == 0.0
        distances = distances + mask * eps
        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1 # torch.eye(): Returns a 2-D tensor with ones on the diagonal and zeros elsewhere

    # Check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = indices_not_equal * labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1

    return mask

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j.byte() * (i_equal_k.byte() ^ 1) ###

    mask = distinct_indices * valid_labels   # Combine the two masks

    return mask

"""## Model"""

class spoof_classifier(nn.Module):
    def __init__(self): # positive parameter means grl or not, True: not grl
        super(spoof_classifier, self).__init__()
        self.shared_encoder_pred_class = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            # nn.Linear(512, 3),
            #nn.Sigmoid() # pred class prob
        )
        self.class_criterion = AngularPenaltySMLoss(512, 3, loss_type='sphereface') # loss_type in ['arcface', 'sphereface', 'cosface']

    def forward(self, x, labels, positive):
        x = self.shared_encoder_pred_class(x)
        if positive == True:
            features, result = self.class_criterion(x, labels, True)
            return features, result
        else:
            features = self.class_criterion(x, labels, False)
        return features

class domain_classifier(nn.Module):
    def __init__(self):
        super(domain_classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )
    def forward(self, x):
        x = self.model(x)
        return x

class decoder(nn.Module):
    def __init__(self, code_size=512):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(3000, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1)
        )
    def forward(self, x):
        x = self.layer(x)
        return x

class depth_decoder(nn.Module):
    def __init__(self, code_size=512):
        super(depth_decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(1000, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),


            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1)
        )
    def forward(self, x):
        x = self.layer(x)
        return x
        
class GRL(torch.autograd.Function):
    def __init__(self):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 4000  # be same to the max_iter of config.py

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput

domain_a_encoder = torchvision.models.resnet18(pretrained=True).to(device)
domain_b_encoder = torchvision.models.resnet18(pretrained=True).to(device)
domain_c_encoder = torchvision.models.resnet18(pretrained=True).to(device)
shared_content = torchvision.models.resnet18(pretrained=True).to(device)
shared_spoof = torchvision.models.resnet18(pretrained=True).to(device)
spoof_classify = spoof_classifier().to(device)
domain_classify = domain_classifier().to(device)
decode = decoder().to(device)
depth_map = depth_decoder().to(device)

"""## Training"""

alpha, beta, gamma = 0.0001, 0.0001, 0.001
#alpha for spoofing  classify MSE to content and domain
#gamma for else 
lr = 0.0003
n_epoch = 100
test_best_auc = 0.0
test_best_acc = 0.0
test_best_hter = 0.0

len_dataloader = min(len(domain1_loader), len(domain2_loader), len(domain3_loader))

opt_domain_a_encoder = optim.AdamW(domain_a_encoder.parameters(), lr = lr)
opt_domain_b_encoder = optim.AdamW(domain_b_encoder.parameters(), lr = lr)
opt_domain_c_encoder = optim.AdamW(domain_c_encoder.parameters(), lr = lr)
opt_shared_content = optim.AdamW(shared_content.parameters(), lr = lr)
opt_shared_spoof = optim.AdamW(shared_spoof.parameters(), lr = lr)
opt_spoof_classify = optim.AdamW(spoof_classify.parameters(), lr = lr)
opt_domain_classify = optim.AdamW(domain_classify.parameters(), lr = lr)
opt_decode = optim.AdamW(decode.parameters(), lr = lr)
opt_depth = optim.AdamW(depth_map.parameters(), lr = lr)
softmax = nn.Softmax(dim=1)
class_criterion = nn.CrossEntropyLoss()
class_criterion_re = MSE()
mse_loss = MSE()
simse_loss = SIMSE()
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

#plot acc
plot_domain = []
plot_spoof = []
plot_content = []
plot_auc = []
plot_acc = []
plot_hter = []

print('epoch num = ', n_epoch, ', iter num = ', len_dataloader)

for epoch in range(n_epoch):
    domain1_loader = DataLoader(domain1_real_dataset + domain1_print_dataset + domain1_replay_dataset, batch_size = batch_size, shuffle = True)
    domain2_loader = DataLoader(domain2_real_dataset + domain2_print_dataset + domain2_replay_dataset, batch_size = batch_size, shuffle = True)
    domain3_loader = DataLoader(domain3_real_dataset + domain3_print_dataset + domain3_replay_dataset, batch_size = batch_size, shuffle = True)
    print('-------------------------------------------------- epoch = {} --------------------------------------------------'.format(str(epoch))) 
    # print('-------------------------------------------------- Oulu Auc = {} --------------------------------------------------'.format(str(oulu_best_auc)))
    # print('-------------------------------------------------- Oulu Acc = {} --------------------------------------------------'.format(str(oulu_best_acc)))
    print('-------------------------------------------------- {} Auc = {} --------------------------------------------------'.format(target_domain, str(test_best_auc)))
    print('-------------------------------------------------- {} Acc = {} --------------------------------------------------'.format(target_domain, str(test_best_acc))) 
    print('-------------------------------------------------- {} Hter = {} --------------------------------------------------'.format(target_domain, str(test_best_hter))) 


    e_domain_class_loss = 0.0 
    e_domain_grl_spoof_loss = 0.0 
    e_domain_grl_content_loss = 0.0 
    e_spoof_class_loss = 0.0 
    e_spoof_grl_content_domain_loss = 0.0 
    e_triplet_loss = 0.0 
    e_recon_loss = 0.0 
    e_content_id_loss = 0.0

    
    for i, ((d1_data, d1_depth, d1_label), (d2_data, d2_depth, d2_label), (d3_data, d3_depth, d3_label)) in enumerate(zip(domain1_loader, domain2_loader, domain3_loader)):

        domain_a_encoder.train()
        domain_b_encoder.train()
        domain_c_encoder.train()
        shared_content.train()
        shared_spoof.train()
        spoof_classify.train()
        domain_classify.train()
        decode.train()
        depth_map.train()

        ###Set iter loss###
        domain_class_loss = 0.0 
        domain_grl_spoof_loss = 0.0 
        domain_grl_content_loss = 0.0 
        spoof_class_loss = 0.0 
        spoof_grl_content_domain_loss = 0.0 
        recon_loss = 0.0 
        depth_loss = 0.0

        ###Set data###
        d1_data = d1_data.expand(len(d1_data), 3, img_size , img_size)[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d2_data = d2_data.expand(len(d2_data), 3, img_size , img_size)[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d3_data = d3_data.expand(len(d3_data), 3, img_size , img_size)[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d1_depth = d1_depth[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d2_depth = d2_depth[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d3_depth = d3_depth[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d1_label = d1_label[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d2_label = d2_label[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d3_label = d3_label[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
   
        # 把所有不同domain的資料混在一起
        mixed_data = torch.cat([d1_data, d2_data, d3_data], dim = 0).to(device)
        mixed_depth = torch.cat([d1_depth, d2_depth, d3_depth], dim = 0).to(device)
        mixed_label = torch.cat([d1_label, d2_label, d3_label], dim = 0).to(device)

        mixed_label_domain = torch.tensor([1/3]).repeat(len(d1_data) + len(d2_data) + len(d3_data), 3).to(device)
        mixed_label_re = torch.tensor([1/3]).repeat(len(d1_data) + len(d2_data) + len(d3_data), 3).to(device) # real, print, replay，要讓模型無法分出來

        #設定domain label
        domain_label_true = torch.zeros([len(d1_data) + len(d2_data) + len(d3_data)],dtype=torch.long).to(device)
        domain_label_true[len(d1_data):len(d1_data) + len(d2_data)] = 1
        domain_label_true[len(d2_data) + len(d3_data):] = 2
   
        ###Extract feature###
        spoof_feature = shared_spoof(mixed_data)
        content_feature = shared_content(mixed_data)
        domain1_feature = domain_a_encoder(d1_data)
        domain2_feature = domain_b_encoder(d2_data)
        domain3_feature = domain_c_encoder(d3_data)
        domain_feature = torch.cat([domain1_feature, domain2_feature, domain3_feature], dim = 0).to(device)

        # ###Step 1 : 訓練 Domain Classifier(正向訓練)###
        domain_logit = domain_classify(domain_feature)
        loss_domain = class_criterion(domain_logit, domain_label_true)

        # total_d = 0
        # correct_d = 0
        # output = domain_logit.view(-1, 3)
        # pred = torch.max(output, 1)[1]
        # for j in range(len(pred)):
        #     if pred[j] == domain_label_true[j]:
        #         correct_d += 1
        #     total_d += 1

        loss = loss_domain
        domain_class_loss += loss 
        e_domain_class_loss += domain_class_loss
        loss.backward()

        opt_domain_a_encoder.step()
        opt_domain_b_encoder.step()
        opt_domain_c_encoder.step()
        opt_domain_classify.step()

        domain_classify.eval() # 不要動
        opt_domain_a_encoder.zero_grad() 
        opt_domain_b_encoder.zero_grad() 
        opt_domain_c_encoder.zero_grad()
        opt_domain_classify.zero_grad() 


        ###Step 2 : 讓Domain Classify GRL回spoof和content###
        #spoof部分
        spoof_domain_logit = softmax(domain_classify(spoof_feature))
        loss_domain = class_criterion_re(spoof_domain_logit, mixed_label_domain) 

        # total_s = 0
        # correct_s = 0
        # output = spoof_domain_logit.view(-1, 3)
        # pred = torch.max(output, 1)[1]
        # for j in range(len(pred)):
        #     if pred[j] == domain_label_true[j]:
        #         correct_s += 1
        #     total_s += 1


        spoof_loss = gamma*loss_domain
        domain_grl_spoof_loss += spoof_loss 
        e_domain_grl_spoof_loss += domain_grl_spoof_loss
        spoof_loss.backward()
        opt_shared_spoof.step()
        opt_shared_spoof.zero_grad() 

        #content部分
        content_domain_logit = softmax(domain_classify(content_feature))
        loss_domain = class_criterion_re(content_domain_logit, mixed_label_domain) 

        # total_c = 0
        # correct_c = 0
        # output = content_domain_logit.view(-1, 3)
        # pred = torch.max(output, 1)[1]
        # for j in range(len(pred)):
        #     if pred[j] == domain_label_true[j]:
        #         correct_c += 1
        #     total_c += 1

        # plot_domain.append(correct_d/total_d)
        # plot_spoof.append(correct_s/total_s)
        # plot_content.append(correct_c/total_c)
        # print("\r Domain acc={:.4f}, Spoof acc={:.4f}, Content acc={:.4f}".format(correct_d/total_d, correct_s/total_s, correct_c/total_c), end = "")

        content_loss = gamma*loss_domain
        domain_grl_content_loss += content_loss 
        e_domain_grl_content_loss += domain_grl_content_loss
        content_loss.backward()
        opt_shared_content.step()
        opt_shared_content.zero_grad() 

        ###Step 3 : 訓練 Spoof Classify(正向訓練)###
        spoof_feature = shared_spoof(mixed_data) 
        # spoof_logit = spoof_classify(spoof_feature)
        _, loss_class = spoof_classify(spoof_feature, mixed_label, True)
        loss_class *= beta
        # loss_class = class_criterion(spoof_logit, mixed_label)
        spoof_class_loss += loss_class 
        e_spoof_class_loss += spoof_class_loss
        loss_class.backward()
        opt_shared_spoof.step()
        opt_spoof_classify.step()
        opt_shared_spoof.zero_grad() 
        opt_spoof_classify.zero_grad() 
        spoof_classify.eval()

        ###Step 4 : 讓Spoof Classify GRL回content和domain###
        content_feature = shared_content(mixed_data) 
        domain1_feature = domain_a_encoder(d1_data)
        domain2_feature = domain_b_encoder(d2_data)
        domain3_feature = domain_b_encoder(d3_data)
        domain_feature = torch.cat([domain1_feature, domain2_feature, domain3_feature], dim = 0).to(device)

        content_logit = softmax(spoof_classify(content_feature, mixed_label_re, False))
        domain_logit = softmax(spoof_classify(domain_feature, mixed_label_re, False))
        loss_domain = class_criterion_re(content_logit, mixed_label_re)
        loss_content = class_criterion_re(domain_logit, mixed_label_re)
        loss = gamma*loss_content + gamma*loss_domain
        spoof_grl_content_domain_loss += loss
        e_spoof_grl_content_domain_loss += spoof_grl_content_domain_loss
        loss.backward()
        opt_shared_content.step()
        opt_domain_a_encoder.step()
        opt_domain_b_encoder.step()
        opt_domain_c_encoder.step()
        opt_shared_content.zero_grad() 
        opt_domain_a_encoder.zero_grad() 
        opt_domain_b_encoder.zero_grad() 
        opt_domain_c_encoder.zero_grad() 

        ###Step 3.5 : 訓練 depth###
        content_feature = shared_content(mixed_data).view(-1, 1000, 1, 1) ###
        depth_recon = depth_map(content_feature)

        err_sim1 = mse_loss(depth_recon, mixed_depth)
        err_sim2 = simse_loss(depth_recon, mixed_depth)
        err = 0.01*err_sim1 + 0.01*err_sim2
        depth_loss += err

        depth_loss.backward()
        opt_shared_content.step()
        opt_depth.step()
        opt_shared_content.zero_grad()
        opt_depth.zero_grad()
        depth_map.eval()

        ###Step 6 : Recon###
        spoof_feature = shared_spoof(mixed_data) 
        content_feature = shared_content(mixed_data) 
        domain1_feature = domain_a_encoder(d1_data)
        domain2_feature = domain_b_encoder(d2_data)
        domain3_feature = domain_c_encoder(d2_data)

        d1_spoof = spoof_feature[:][:len(d1_data)]
        d2_spoof = spoof_feature[:][len(d1_data):len(d1_data) + len(d2_data)]
        d3_spoof = spoof_feature[:][len(d2_data) + len(d3_data):]

        d1_content = content_feature[:][:len(d1_data)]
        d2_content = content_feature[:][len(d1_data):len(d1_data) + len(d2_data)]
        d3_content = content_feature[:][len(d2_data) + len(d3_data):]

        d1_recon = torch.cat([d1_spoof, d1_content, domain1_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        d2_recon = torch.cat([d2_spoof, d2_content, domain2_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        d3_recon = torch.cat([d3_spoof, d3_content, domain3_feature], dim = 1).view(-1, 3000, 1, 1).to(device)

        d1_recon = decode(d1_recon)
        d2_recon = decode(d2_recon)
        d3_recon = decode(d3_recon)

        err_sim1 = mse_loss(d1_recon, d1_data)
        err_sim2 = simse_loss(d1_recon, d1_data)
        err_sim3 = mse_loss(d2_recon, d2_data)
        err_sim4 = simse_loss(d2_recon, d2_data)
        err_sim5 = mse_loss(d3_recon, d3_data)
        err_sim6 = simse_loss(d3_recon, d3_data)
        
        err = 0.01*err_sim1 + 0.01*err_sim2 + 0.01*err_sim3 + 0.01*err_sim4 + 0.01*err_sim5 + 0.01*err_sim6
        recon_loss += err
        e_recon_loss += recon_loss
        err.backward()

        opt_shared_spoof.step()
        opt_shared_content.step()
        opt_domain_a_encoder.step()
        opt_domain_b_encoder.step()
        opt_domain_c_encoder.step()
        opt_decode.step()
        opt_shared_spoof.zero_grad() 
        opt_shared_content.zero_grad() 
        opt_domain_a_encoder.zero_grad() 
        opt_domain_b_encoder.zero_grad() 
        opt_domain_c_encoder.zero_grad() 
        opt_decode.zero_grad() 

        ''' feature disentanglement '''
        domain_classify.eval()
        spoof_classify.eval()
        d1_data = d1_data.expand(len(d1_data), 3, img_size , img_size)[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d2_data = d2_data.expand(len(d2_data), 3, img_size , img_size)[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d3_data = d3_data.expand(len(d3_data), 3, img_size , img_size)[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d1_label = d1_label[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d2_label = d2_label[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)
        d3_label = d3_label[:min(len(d1_data), len(d2_data), len(d3_data))].to(device)

        domain_label_true = torch.zeros([(len(d1_data) + len(d2_data) + len(d3_data))*2],dtype=torch.long).to(device)
        domain_label_true[len(d1_data)*2:(len(d1_data) + len(d2_data))*2] = 1
        domain_label_true[(len(d2_data) + len(d3_data))*2:] = 2

        mixed_data = torch.cat([d1_data, d2_data, d3_data], dim = 0).to(device)
        spoof_feature = shared_spoof(mixed_data)
        spoof1_feature = spoof_feature[:len(d1_data)]
        spoof2_feature = spoof_feature[len(d1_data):len(d1_data) + len(d2_data)]
        spoof3_feature = spoof_feature[len(d2_data) + len(d3_data):]
        content_feature = shared_content(mixed_data)
        content1_feature = content_feature[:len(d1_data)]
        content2_feature = content_feature[len(d1_data):len(d1_data) + len(d2_data)]
        content3_feature = content_feature[len(d2_data) + len(d3_data):]
        domain1_feature = domain_a_encoder(d1_data)
        domain2_feature = domain_b_encoder(d2_data)
        domain3_feature = domain_c_encoder(d3_data)

        ###for domain###

        d1to2_recon = torch.cat([spoof1_feature, content1_feature, domain2_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        d1to3_recon = torch.cat([spoof1_feature, content1_feature, domain3_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        d2to1_recon = torch.cat([spoof2_feature, content2_feature, domain1_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        d2to3_recon = torch.cat([spoof2_feature, content2_feature, domain3_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        d3to1_recon = torch.cat([spoof3_feature, content3_feature, domain1_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        d3to2_recon = torch.cat([spoof3_feature, content3_feature, domain2_feature], dim = 1).view(-1, 3000, 1, 1).to(device)

        d1to2_recon = decode(d1to2_recon)
        d1to3_recon = decode(d1to3_recon)
        d2to1_recon = decode(d2to1_recon)
        d2to3_recon = decode(d2to3_recon)
        d3to1_recon = decode(d3to1_recon)
        d3to2_recon = decode(d3to2_recon)

        d1to2_recon_feature = domain_b_encoder(d1to2_recon)
        d1to3_recon_feature = domain_c_encoder(d1to3_recon)
        d2to1_recon_feature = domain_a_encoder(d2to1_recon)
        d2to3_recon_feature = domain_c_encoder(d2to3_recon)
        d3to1_recon_feature = domain_a_encoder(d3to1_recon)
        d3to2_recon_feature = domain_b_encoder(d3to2_recon)

        domain_recon_feature = torch.cat([d2to1_recon_feature, d3to1_recon_feature, d1to2_recon_feature, 
                                          d3to2_recon_feature, d1to3_recon_feature, d2to3_recon_feature], dim = 0).to(device)
        domain_recon_logit = domain_classify(domain_recon_feature)
        loss_swap_domain = class_criterion(domain_recon_logit, domain_label_true)

        ###for spoof###
        s1to2_recon = torch.cat([spoof2_feature, content1_feature, domain1_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        s1to3_recon = torch.cat([spoof3_feature, content1_feature, domain1_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        s2to1_recon = torch.cat([spoof1_feature, content2_feature, domain2_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        s2to3_recon = torch.cat([spoof3_feature, content2_feature, domain2_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        s3to1_recon = torch.cat([spoof1_feature, content3_feature, domain3_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        s3to2_recon = torch.cat([spoof2_feature, content3_feature, domain3_feature], dim = 1).view(-1, 3000, 1, 1).to(device)
        s1to2_recon = decode(s1to2_recon)
        s1to3_recon = decode(s1to3_recon)
        s2to1_recon = decode(s2to1_recon)
        s2to3_recon = decode(s2to3_recon)
        s3to1_recon = decode(s3to1_recon)
        s3to2_recon = decode(s3to2_recon)
        s_recon_feature = shared_spoof(torch.cat([s2to1_recon, s3to1_recon, s1to2_recon,
                                                  s3to2_recon, s1to3_recon, s2to3_recon], dim = 0)).to(device)
        # spoof_recon_logit = spoof_classify(s_recon_feature)
        mixed_label = torch.cat([d1_label, d1_label, d2_label, d2_label, d3_label, d3_label], dim = 0).to(device)
        _, loss_swap_spoof = spoof_classify(s_recon_feature, mixed_label, True)

        swap_loss = lambda_function(10, epoch)*(loss_swap_domain + beta*loss_swap_spoof)
        swap_loss.backward() 
        opt_shared_spoof.step()
        opt_domain_a_encoder.step()
        opt_domain_b_encoder.step()
        opt_domain_c_encoder.step()
        opt_shared_spoof.zero_grad() 
        opt_domain_a_encoder.zero_grad() 
        opt_domain_b_encoder.zero_grad()
        opt_domain_c_encoder.zero_grad()

        domain1_real_loader = DataLoader(domain1_real_dataset, batch_size = batch_size, shuffle = True)
        domain1_print_loader = DataLoader(domain1_print_dataset, batch_size = batch_size, shuffle = True)
        domain1_replay_loader = DataLoader(domain1_replay_dataset, batch_size = batch_size, shuffle = True)
        domain2_real_loader = DataLoader(domain2_real_dataset, batch_size = batch_size, shuffle = True)
        domain2_print_loader = DataLoader(domain2_print_dataset, batch_size = batch_size, shuffle = True)
        domain2_replay_loader = DataLoader(domain2_replay_dataset, batch_size = batch_size, shuffle = True)
        domain3_real_loader = DataLoader(domain3_real_dataset, batch_size = batch_size, shuffle = True)
        domain3_print_loader = DataLoader(domain3_print_dataset, batch_size = batch_size, shuffle = True)
        domain3_replay_loader = DataLoader(domain3_replay_dataset, batch_size = batch_size, shuffle = True)

        ''' triplet loss '''
        for k, ((d1_real, _, d1_real_label), (d1_print, _, d1_print_label), (d1_replay, _, d1_replay_label), 
                (d2_real, _, d2_real_label), (d2_print, _, d2_print_label), (d2_replay, _, d2_replay_label), 
                (d3_real, _, d3_real_label), (d3_print, _, d3_print_label), (d3_replay, _, d3_replay_label)) in enumerate( \
            zip(domain1_real_loader, domain1_print_loader, domain1_replay_loader, \
                domain2_real_loader, domain2_print_loader, domain2_replay_loader, \
                domain3_real_loader, domain3_print_loader, domain3_replay_loader \
                )):
            # print(len(d1_real), len(d1_print), len(d1_replay), len(d1_real), len(d1_print), len(d1_replay), len(d1_real), len(d1_print), len(d1_replay))
            data_len = min(len(d1_real), len(d1_print), len(d1_replay), len(d1_real), len(d1_print), len(d1_replay), len(d1_real), len(d1_print), len(d1_replay))
            mixed_data = torch.cat([d1_real[:data_len], d1_print[:data_len], d1_replay[:data_len], 
                                    d2_real[:data_len], d2_print[:data_len], d2_replay[:data_len], 
                                    d3_real[:data_len], d3_print[:data_len], d3_replay[:data_len]], dim = 0).to(device)
            spoof_feature = shared_spoof(mixed_data)
            d1_real = spoof_feature[:data_len]
            d1_print = spoof_feature[data_len:data_len*2]
            d1_replay = spoof_feature[data_len*2:data_len*3]
            d2_real = spoof_feature[data_len*3:data_len*4]
            d2_print = spoof_feature[data_len*4:data_len*5]
            d2_replay = spoof_feature[data_len*5:data_len*6]
            d3_real = spoof_feature[data_len*6:data_len*7]
            d3_print = spoof_feature[data_len*7:data_len*8]
            d3_replay = spoof_feature[data_len*8:]
            spoof_triplet_loss = sample_triplet(triplet_loss, d1_real, d1_print, d1_replay, d2_real, d2_print, d2_replay, d3_real, d3_print, d3_replay) 
            print(spoof_triplet_loss)
            spoof_triplet_loss.backward()
            opt_shared_spoof.step()
            opt_shared_spoof.zero_grad()

        print("\r {}/{} domain_class_loss:{:.7f}, depth_loss = {:.7f}, domain_grl_spoof_loss={:.7f}, domain_grl_content_loss={:.7f}, spoof_class_loss={:.4f}, spoof_grl_content_domain_loss={:.7f}, recon_loss = {:.7f}, swap_loss = {:.7f}".format(
                i+1, len_dataloader, domain_class_loss.item(), depth_loss.item(), domain_grl_spoof_loss.item() , domain_grl_content_loss.item(), spoof_class_loss.item(), 
                spoof_grl_content_domain_loss.item(), recon_loss.item(), swap_loss.item()), end = "")

    shared_spoof.eval()
    spoof_classify.eval()
    ans = []
    pred = []
    correct = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            print("\r", batch_idx, '/', len(test_loader), end = "")
            im, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)

            result = shared_spoof(im)
            features, loss = spoof_classify(result, label, True)
            print('spoof_class_loss={:.4f}'.format(loss))
            for j in range(len(features)):
                if label[j].item() == 0:
                    ans.append(1)
                else:
                    ans.append(0)
                pred.append(softmax(features)[j][0].item())
                if label[j].item() == torch.argmax(features[j], dim=0).item():
                    correct += 1
    test_auc = roc_auc_score(ans, pred)
    _, test_hter = HTER(np.array(pred), np.array(ans))
    print('Final {} test auc = {}'.format(target_domain, test_auc))
    print('Final {} test acc = {}'.format(target_domain, correct/len(test_dataset)))
    print('Final {} test hter = {}'.format(target_domain, test_hter))
    plot_auc.append(test_auc)
    plot_acc.append(test_auc)
    plot_hter.append(test_hter)
    if test_auc >= test_best_auc:
        test_best_auc = test_auc
        test_best_acc = correct/len(test_dataset)
        test_best_hter = test_hter
        torch.save(shared_spoof, shared_spoof_path)
        torch.save(spoof_classify, spoof_classify_path)
        torch.save(shared_content, shared_content_path)
        torch.save(depth_map, depth_map_path)
        torch.save(domain_a_encoder, domain1_encoder_path)
        torch.save(domain_b_encoder, domain2_encoder_path)
        torch.save(domain_c_encoder, domain3_encoder_path)
        torch.save(domain_classify, domain_classify_path)
        torch.save(decode, decoder_path)
        print('{}: save model'.format(target_domain))

if test_auc >= test_best_auc:
        test_best_auc = test_auc
        test_best_acc = correct/len(test_dataset)
        torch.save(shared_spoof, shared_spoof_path)
        torch.save(spoof_classify, spoof_classify_path)
        torch.save(shared_content, shared_content_path)
        torch.save(depth_map, depth_map_path)
        torch.save(domain_a_encoder, domain1_encoder_path)
        torch.save(domain_b_encoder, domain2_encoder_path)
        torch.save(domain_c_encoder, domain3_encoder_path)
        torch.save(domain_classify, domain_classify_path)
        torch.save(decode, decoder_path)
        print('{}: save model'.format(target_domain))

# Ethen
shared_spoof = torch.load(os.path.join(path, 'model/4/oulu/oulu_spoof_encoder.pt'), map_location=device)
spoof_classify = torch.load(os.path.join(path, 'model/4/oulu/oulu_spoof_classify.pt'), map_location=device)
# shared_spoof = torch.load(os.path.join(path, 'model/4/casia/casia_spoof_encoder.pt'), map_location=device)
# spoof_classify = torch.load(os.path.join(path, 'model/4/casia/casia_spoof_classify.pt'), map_location=device)

shared_spoof.eval()
spoof_classify.eval()

softmax = nn.Softmax(dim=1)
ans = []
pred = []
correct = 0
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        print("\r", batch_idx, '/', len(test_loader))
        im, label = data
        im, label = im.to(device), label.to(device)
        im = im.expand(im.data.shape[0], 3, 256, 256)
        result = shared_spoof(im)
        features, loss  = spoof_classify(result, label, True)
        for i in range(len(features)):
            if label[i].item() == 0:
                ans.append(1)
            else:
                ans.append(0)
            pred.append(softmax(features)[i][0].item())
            # pred.append(features[i][0].item())
            if label[i].item() == torch.argmax(features[i], dim=0).item():
                correct += 1

test_auc = roc_auc_score(ans, pred)
_, test_hter = HTER(np.array(pred), np.array(ans))
print('Final {} test auc = {}'.format(target_domain, test_auc))
print('Final {} test acc = {}'.format(target_domain, correct/len(test_dataset)))
print('Final {} test hter = {}'.format(target_domain, test_hter))

plt.figure(figsize=(10, 5))
plt.title("Auc/Acc/Hter During Epoch")
plt.plot(plot_auc)
plt.plot(plot_acc)
plt.plot(plot_hter)
plt.xlabel('Epoch')
plt.ylabel("Value")
plt.show()