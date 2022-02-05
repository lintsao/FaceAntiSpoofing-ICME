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

from loss import *

"""## Model"""
class share_feature(nn.Module): # 2 types: real, fake
    def __init__(self): # positive parameter means grl or not, True: not grl
        super(share_feature, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
        )
        #self.class_criterion = AngularPenaltySMLoss(512, 2, loss_type='cosface') # loss_type in ['arcface', 'sphereface', 'cosface']

    def forward(self, x):
        x = self.cnn_layers(x)
        return x
class spoof_classifier_acc(nn.Module): # 3 types: real, print, replay
    def __init__(self): # positive parameter means grl or not, True: not grl
        super(spoof_classifier_acc, self).__init__()
        self.shared_encoder_pred_class = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
        )
        self.class_criterion = AngularPenaltySMLoss(512, 3, loss_type='cosface') # loss_type in ['arcface', 'sphereface', 'cosface']

    def forward(self, x, labels, positive):
        x = self.shared_encoder_pred_class(x)
        if positive == True:
            features, result = self.class_criterion(x, labels, True)
            return features, result
        else:
            features = self.class_criterion(x, labels, False)
        return features

class spoof_classifier_auc(nn.Module): # 2 types: real, fake
    def __init__(self): # positive parameter means grl or not, True: not grl
        super(spoof_classifier_auc, self).__init__()
        self.shared_encoder_pred_class = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
        )
        self.class_criterion = AngularPenaltySMLoss(512, 2, loss_type='cosface') # loss_type in ['arcface', 'sphereface', 'cosface']

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