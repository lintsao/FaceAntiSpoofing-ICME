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

def test_auc(args):
    same_seeds(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if use_cuda else "cpu")
    print("finish initialization, device: {}".format(device))
    
    # target:MSU    => 1: i, 2: o, 3: c
    # target:Idiap  => 1: m, 2: o, 3: c
    # target:Oulu   => 1: m, 2: i, 3: c
    # target:Casia  => 1: m, 2: i, 3: o

    shared_spoof = torch.load(args.spoof_encoder, map_location=device)
    spoof_classify = torch.load(args.spoof_classifier, map_location=device)
    print("-------------------------------------------------- finish model --------------------------------------------------")

    test_dataset, _, _, _, \
    _, _, _, _, \
    _, _ = choose_dataset(args.dataset_path, args.target_domain, args.img_size, args.depth_size)
    print("test_dataset:{}".format(len(test_dataset)))
    print("-------------------------------------------------- finish dataset --------------------------------------------------")
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, pin_memory=True, num_workers = 4)

    shared_spoof.eval()
    spoof_classify.eval()
    ans = []
    pred = []
    correct = 0
    softmax = nn.Softmax(dim=0)
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            print("\r", batch_idx, '/', len(test_loader), end = "")
            im, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)

            result = shared_spoof(im)
            features, loss = spoof_classify(result, label, True)
            # print('spoof_class_loss={:.4f}'.format(loss))
            for j in range(len(features)):
                if label[j].item() == 0:
                    ans.append(1)
                else:
                    ans.append(0)
            
                prob = softmax(features[j])[0].item()
                pred.append(prob)
                if label[j].item() == torch.argmax(softmax(features[j]), dim=0).item():
                    correct += 1

    test_auc = roc_auc_score(ans, pred)
    test_acc = correct/len(test_dataset)
    _, test_hter = HTER(np.array(pred), np.array(ans))

    print('Final {} test auc = {}'.format(args.target_domain, test_auc))
    print('Final {} test acc = {}'.format(args.target_domain, test_acc))
    print('Final {} test hter = {}'.format(args.target_domain, test_hter))