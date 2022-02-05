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

from utils import *
from model import *
from loss import *
from dataset_auc import *

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