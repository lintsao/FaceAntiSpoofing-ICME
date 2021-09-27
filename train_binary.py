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

def train_binary(args):
    same_seeds(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("finish initialization, device: {}".format(device))
    
    # target:MSU    => 1: i, 2: o, 3: c
    # target:Idiap  => 1: m, 2: o, 3: c
    # target:Oulu   => 1: m, 2: i, 3: c
    # target:Casia  => 1: m, 2: i, 3: o

    shared_spoof_path, spoof_classify_path, shared_content_path, \
    domain1_encoder_path, domain2_encoder_path, domain3_encoder_path, \
    domain_classify_path, decoder_path, depth_map_path = make_model_path(args.path, args.target_domain, args.number_folder) 
    print("-------------------------------------------------- finish model path --------------------------------------------------")

    test_dataset, domain1_real_dataset, domain1_print_dataset, domain1_replay_dataset, \
    domain2_real_dataset, domain2_print_dataset, domain2_replay_dataset, domain3_real_dataset, \
    domain3_print_dataset, domain3_replay_dataset = choose_dataset(args.dataset_path, args.target_domain, args.img_size, args.depth_size)
    print("-------------------------------------------------- finish dataset --------------------------------------------------")

    print("test_dataset:{}".format(len(test_dataset)))
    print("domain1_dataset:{}".format(len(domain1_real_dataset + domain1_print_dataset + domain1_replay_dataset)))
    print("domain2_dataset:{}".format(len(domain2_real_dataset + domain2_print_dataset + domain2_replay_dataset)))
    print("domain3_dataset:{}".format(len(domain3_real_dataset + domain3_print_dataset + domain3_replay_dataset)))

    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
    domain1_loader = DataLoader(domain1_real_dataset + domain1_print_dataset + domain1_replay_dataset, batch_size = args.batch_size, shuffle = True)
    domain2_loader = DataLoader(domain2_real_dataset + domain2_print_dataset + domain2_replay_dataset, batch_size = args.batch_size, shuffle = True)
    domain3_loader = DataLoader(domain3_real_dataset + domain3_print_dataset + domain3_replay_dataset, batch_size = args.batch_size, shuffle = True)

    # shared_spoof = torchvision.models.resnet18(pretrained=True).to(device)
    # shared_spoof = torchvision.models.resnet18(pretrained=False).to(device)
    shared_spoof = torchvision.models.alexnet(pretrained=True).to(device)
    spoof_classify = spoof_classifier_auc().to(device)
    print("-------------------------------------------------- finish model --------------------------------------------------")

    """## Training"""

    alpha, beta_depth, beta_faces, gamma = 0.0001, 0.0001, 0.0001, 0.0001  # beta: spoof, gamma: grl
    #alpha for spoofing  classify MSE to content and domain
    #gamma for else 
    test_best_auc = 0.0
    test_best_acc = 0.0
    test_best_hter = 0.0
    test_best_epoch = 0

    len_dataloader = min(len(domain1_loader), len(domain2_loader), len(domain3_loader))

    opt_shared_spoof = optim.AdamW(shared_spoof.parameters(), lr = args.lr)
    opt_spoof_classify = optim.AdamW(spoof_classify.parameters(), lr = args.lr)

    opt_shared_spoof_scheduler = optim.lr_scheduler.MultiStepLR(opt_shared_spoof, milestones=[30, 70, 90], gamma=0.3)
    opt_spoof_classify_scheduler = optim.lr_scheduler.MultiStepLR(opt_spoof_classify, milestones=[70, 90], gamma=0.3)

    softmax = nn.Softmax(dim=0)

    #plot acc
    plot_auc = []
    plot_acc = []
    plot_hter = []

    print('epoch num = ', args.n_epoch, ', iter num = ', len_dataloader)

    for epoch in range(args.n_epoch):
        domain1_loader = DataLoader(domain1_real_dataset + domain1_print_dataset + domain1_replay_dataset, batch_size = args.batch_size, shuffle = True)
        domain2_loader = DataLoader(domain2_real_dataset + domain2_print_dataset + domain2_replay_dataset, batch_size = args.batch_size, shuffle = True)
        domain3_loader = DataLoader(domain3_real_dataset + domain3_print_dataset + domain3_replay_dataset, batch_size = args.batch_size, shuffle = True)
        print('-------------------------------------------------- epoch = {} --------------------------------------------------'.format(str(epoch))) 
        print('-------------------------------------------------- {} Auc = {} --------------------------------------------------'.format(args.target_domain, str(test_best_auc)))
        print('-------------------------------------------------- {} Acc = {} --------------------------------------------------'.format(args.target_domain, str(test_best_acc))) 
        print('-------------------------------------------------- {} Hter = {} --------------------------------------------------'.format(args.target_domain, str(test_best_hter))) 
        print('-------------------------------------------------- {} @epoch = {} --------------------------------------------------'.format(args.target_domain, str(test_best_epoch))) 

        e_spoof_class_loss = 0.0 
        
        for i, ((d1_data, d1_depth, d1_label), (d2_data, d2_depth, d2_label), (d3_data, d3_depth, d3_label)) in enumerate(zip(domain1_loader, domain2_loader, domain3_loader)):

            shared_spoof.train()
            spoof_classify.train()

            ###Set iter loss###
            spoof_class_loss = 0.0 

            ###Set data###
            len_data = min(len(d1_data), len(d2_data), len(d3_data))
            d1_data = d1_data.expand(len(d1_data), 3, args.img_size , args.img_size)[:len_data].to(device)
            d2_data = d2_data.expand(len(d2_data), 3, args.img_size , args.img_size)[:len_data].to(device)
            d3_data = d3_data.expand(len(d3_data), 3, args.img_size , args.img_size)[:len_data].to(device)
            d1_depth = d1_depth[:len_data].to(device)
            d2_depth = d2_depth[:len_data].to(device)
            d3_depth = d3_depth[:len_data].to(device)
            d1_label = d1_label[:len_data].to(device)
            d2_label = d2_label[:len_data].to(device)
            d3_label = d3_label[:len_data].to(device)
    
            # 把所有不同domain的資料混在一起
            mixed_data = torch.cat([d1_data, d2_data, d3_data], dim = 0).to(device)
            mixed_label = torch.cat([d1_label, d2_label, d3_label], dim = 0).to(device)

            ###Extract feature###
            spoof_feature = shared_spoof(mixed_data)

            ###Step 3 : 訓練 Spoof Classify(正向訓練)###
            spoof_feature = shared_spoof(mixed_data) 
            _, spoof_class_loss = spoof_classify(spoof_feature, mixed_label, True)
            # spoof_class_loss *= beta
            e_spoof_class_loss += spoof_class_loss

            loss = spoof_class_loss
            loss.backward()
            opt_shared_spoof.step()
            opt_spoof_classify.step()

            opt_shared_spoof.zero_grad() 
            opt_spoof_classify.zero_grad() 
            spoof_classify.eval()

            print("\r {}/{} spoof_class_loss={:.4f}".format(
                    i+1, len_dataloader, spoof_class_loss.item(), 
                    ), end = "")
        
        opt_shared_spoof_scheduler.step()
        opt_spoof_classify_scheduler.step()

        print("{} lr: {}".format(epoch, opt_spoof_classify_scheduler.get_last_lr()[0]))

        print("{}/{} e_spoof_class_loss={:.4f}".format(
                i+1, args.n_epoch, e_spoof_class_loss.item(), 
                ))

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
        print(pred)
        test_auc = roc_auc_score(ans, pred)
        test_acc = correct/len(test_dataset)
        _, test_hter = HTER(np.array(pred), np.array(ans))

        print('Final {} test auc = {}'.format(args.target_domain, test_auc))
        print('Final {} test acc = {}'.format(args.target_domain, test_acc))
        print('Final {} test hter = {}'.format(args.target_domain, test_hter))

        plot_auc.append(test_auc)
        plot_acc.append(test_acc)
        plot_hter.append(test_hter)
        # if test_auc > test_best_auc:
        #     test_best_auc = test_auc
        #     test_best_acc = test_acc
        #     test_best_hter = test_hter
        #     test_best_epoch = epoch
        torch.save(shared_spoof, shared_spoof_path)
        torch.save(spoof_classify, spoof_classify_path)
        print('{}: save model'.format(args.target_domain))