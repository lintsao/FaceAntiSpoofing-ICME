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
import sys

from model import *

"""## Utils"""

def same_seeds(seed=307):
    print("seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_model_path(path, target_domain, number_folder):
    print("path: {}, target_domain: {}, number_folder: {}".format(path, target_domain, number_folder))
    if not os.path.exists(os.path.join(path, 'model', target_domain)):
        os.makedirs(os.path.join(path, 'model', target_domain))

    if not os.path.exists(os.path.join(path, 'model', target_domain, number_folder)):
        os.makedirs(os.path.join(path, 'model', target_domain, number_folder))

    shared_spoof_path = os.path.join(path, 'model', target_domain, number_folder, '{}_spoof_encoder.pt'.format(target_domain))
    spoof_classify_path = os.path.join(path, 'model', target_domain, number_folder, '{}_spoof_classify.pt'.format(target_domain))
    shared_content_path = os.path.join(path, 'model', target_domain, number_folder, '{}_content_encoder.pt'.format(target_domain))
    if target_domain == 'msu':           
        domain1_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_i_domain_encoder.pt'.format(target_domain))
        domain2_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_o_domain_encoder.pt'.format(target_domain))
        domain3_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_c_domain_encoder.pt'.format(target_domain))
    elif target_domain == 'idiap':
        domain1_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_m_domain_encoder.pt'.format(target_domain))
        domain2_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_o_domain_encoder.pt'.format(target_domain))
        domain3_encoder_path = os.path.join(path, 'model', target_domain, number_folder,'{}_c_domain_encoder.pt'.format(target_domain))
    elif target_domain == 'oulu':
        domain1_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_m_domain_encoder.pt'.format(target_domain))
        domain2_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_i_domain_encoder.pt'.format(target_domain))
        domain3_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_c_domain_encoder.pt'.format(target_domain))
    elif target_domain == 'casia':
        domain1_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_m_domain_encoder.pt'.format(target_domain))
        domain2_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_i_domain_encoder.pt'.format(target_domain))
        domain3_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_o_domain_encoder.pt'.format(target_domain))
    domain_classify_path = os.path.join(path, 'model', target_domain, number_folder, '{}_domain_classify.pt'.format(target_domain))
    decoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_decoder.pt'.format(target_domain))
    depth_map_path = os.path.join(path, 'model', target_domain, number_folder, '{}_depth_map.pt'.format(target_domain))
    return shared_spoof_path, spoof_classify_path, shared_content_path, domain1_encoder_path, domain2_encoder_path, domain3_encoder_path, \
    domain_classify_path, decoder_path, depth_map_path

def load_model_path(path, target_domain, number_folder, scratch, device):
    if scratch == 'True':
        print("use new model")
        domain_a_encoder = torchvision.models.resnet18(pretrained=True).to(device)
        domain_b_encoder = torchvision.models.resnet18(pretrained=True).to(device)
        domain_c_encoder = torchvision.models.resnet18(pretrained=True).to(device)
        shared_content = torchvision.models.resnet18(pretrained=True).to(device)
        shared_spoof = torchvision.models.resnet18(pretrained=True).to(device)
        spoof_classify = spoof_classifier_auc().to(device)
        domain_classify = domain_classifier().to(device)
        depth_map = depth_decoder().to(device)
    else:
        print("use target_domain: {}, number_folder: {} model".format(target_domain, number_folder))
        shared_spoof_path = os.path.join(path, 'model', target_domain, number_folder, '{}_spoof_encoder.pt'.format(target_domain))
        spoof_classify_path = os.path.join(path, 'model', target_domain, number_folder, '{}_spoof_classify.pt'.format(target_domain))
        shared_content_path = os.path.join(path, 'model', target_domain, number_folder, '{}_content_encoder.pt'.format(target_domain))
        if target_domain == 'msu':           
            domain1_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_i_domain_encoder.pt'.format(target_domain))
            domain2_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_o_domain_encoder.pt'.format(target_domain))
            domain3_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_c_domain_encoder.pt'.format(target_domain))
        elif target_domain == 'idiap':
            domain1_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_m_domain_encoder.pt'.format(target_domain))
            domain2_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_o_domain_encoder.pt'.format(target_domain))
            domain3_encoder_path = os.path.join(path, 'model', target_domain, number_folder,'{}_c_domain_encoder.pt'.format(target_domain))
        elif target_domain == 'oulu':
            domain1_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_m_domain_encoder.pt'.format(target_domain))
            domain2_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_i_domain_encoder.pt'.format(target_domain))
            domain3_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_c_domain_encoder.pt'.format(target_domain))
        elif target_domain == 'casia':
            domain1_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_m_domain_encoder.pt'.format(target_domain))
            domain2_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_i_domain_encoder.pt'.format(target_domain))
            domain3_encoder_path = os.path.join(path, 'model', target_domain, number_folder, '{}_o_domain_encoder.pt'.format(target_domain))
        domain_classify_path = os.path.join(path, 'model', target_domain, number_folder, '{}_domain_classify.pt'.format(target_domain))
        depth_map_path = os.path.join(path, 'model', target_domain, number_folder, '{}_depth_map.pt'.format(target_domain))

        domain_a_encoder = torch.load(domain1_encoder_path)
        domain_b_encoder = torch.load(domain2_encoder_path)
        domain_c_encoder = torch.load(domain3_encoder_path)
        shared_content = torch.load(shared_content_path)
        shared_spoof = torch.load(shared_spoof_path)
        spoof_classify = torch.load(spoof_classify_path)
        domain_classify = torch.load(domain_classify_path)
        depth_map = torch.load(depth_map_path)
    return domain_a_encoder, domain_b_encoder, domain_c_encoder, shared_content, shared_spoof, spoof_classify, \
    domain_classify, depth_map

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

def sample_triplet_MADDG(triplet_loss, d1_real, d1_print, d1_replay, d2_real, d2_print, d2_replay, d3_real, d3_print, d3_replay):
    output_loss = 0.0
    anchor_domain = list(np.random.choice(a=[1, 2, 3], size=6, replace=True, p=None))
    positive_domain = []
    negative_domain = []
    for index in range(len(anchor_domain)):
        domain_choice = [1, 2, 3]
        domain_choice.remove(anchor_domain[index])
        positive_domain.append(int(np.random.choice(a=domain_choice, size=1, replace=True, p=None)))
        negative_domain.append(int(np.random.choice(a=domain_choice, size=1, replace=True, p=None)))
        # print(anchor_domain[index], positive_domain[index], negative_domain[index])
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
        # print(triplet_loss(anchor, positive, negative))    
        output_loss += triplet_loss(anchor, positive, negative)
    return output_loss

def sample_triplet_SSDG(triplet_loss, d1_real, d1_print, d1_replay, d2_real, d2_print, d2_replay, d3_real, d3_print, d3_replay):
    output_loss = 0.0
    anchor_domain = list(np.random.choice(a=[1, 2, 3], size=6, replace=True, p=None))
    data_len = int((min(len(d1_real), len(d1_print), len(d1_replay), len(d1_real), len(d1_print), len(d1_replay), len(d1_real), len(d1_print), len(d1_replay)))/2)
    d1_real_1 = d1_real[:data_len]
    d1_real_2 = d1_real[data_len:data_len*2]
    d1_print_1 = d1_print[:data_len]
    d1_print_2 = d1_print[data_len:data_len*2]
    d1_replay_1 = d1_replay[:data_len]
    d1_replay_2 = d1_real[data_len:data_len*2]
    d2_real_1 = d2_real[:data_len]
    d2_real_2 = d2_real[data_len:data_len*2]
    d2_print_1 = d2_print[:data_len]
    d2_print_2 = d2_print[data_len:data_len*2]
    d2_replay_1 = d2_replay[:data_len]
    d2_replay_2 = d2_replay[data_len:data_len*2]
    d3_real_1 = d3_real[:data_len]
    d3_real_2 = d3_real[data_len:data_len*2]
    d3_print_1 = d3_print[:data_len]
    d3_print_2 = d3_print[data_len:data_len*2]
    d3_replay_1 = d3_replay[:data_len]
    d3_replay_2 = d3_replay[data_len:data_len*2]

    positive_domain = []
    negative_domain = []
    for index in range(len(anchor_domain)):
        domain_choice = [1, 2, 3]
        domain_choice.remove(anchor_domain[index])
        positive_domain.append(int(np.random.choice(a=domain_choice, size=1, replace=True, p=None)))
        negative_domain.append(int(np.random.choice(a=domain_choice, size=1, replace=True, p=None)))
        # print(anchor_domain[index], positive_domain[index], negative_domain[index])
        if index == 0:    # rrp # real 都要一樣，不用分 domain
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
                anchor = d1_print_1
                positive = d1_print_2
            elif anchor_domain[index] == 2:
                anchor = d2_print_1
                positive = d2_print_2
            else:
                anchor = d3_print_1
                positive = d3_print_2
            if negative_domain[index] == 1:
                negative = d1_replay_2
            elif negative_domain[index] == 2:
                negative = d2_replay_2
            else:
                negative = d3_replay_2
        elif index == 2:  # RRr
            if anchor_domain[index] == 1:
                anchor = d1_replay_1
                positive = d1_replay_2
            elif anchor_domain[index] == 2:
                anchor = d2_replay_1
                positive = d2_replay_2
            else:
                anchor = d3_replay_1
                positive = d3_replay_2
            if negative_domain[index] == 1:
                negative = d1_real_2
            elif negative_domain[index] == 2:
                negative = d2_real_2
            else:
                negative = d3_real_2
        elif index == 3:  # rrR # real 都要一樣，不用分 domain
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
                anchor = d1_print_1
                positive = d1_print_2
            elif anchor_domain[index] == 2:
                anchor = d2_print_1
                positive = d2_print_2
            else:
                anchor = d3_print_1
                positive = d3_print_2
            if negative_domain[index] == 1:
                negative = d1_real_2
            elif negative_domain[index] == 2:
                negative = d2_real_2
            else:
                negative = d3_real_2
        elif index == 5:  # RRp
            if anchor_domain[index] == 1:
                anchor = d1_replay_1
                positive = d1_replay_2
            elif anchor_domain[index] == 2:
                anchor = d2_replay_1
                positive = d2_replay_2
            else:
                anchor = d3_replay_1
                positive = d3_replay_2
            if negative_domain[index] == 1:
                negative = d1_print_2
            elif negative_domain[index] == 2:
                negative = d2_print_2
            else:
                negative = d3_print_2
        # print((anchor.shape, positive.shape, negative.shape))
        # print(triplet_loss(anchor, positive, negative))    
        output_loss += triplet_loss(anchor, positive, negative)
    return output_loss

def lambda_function(gamma, epoch):
    return 2/(1 + math.exp(-gamma*epoch)) - 1