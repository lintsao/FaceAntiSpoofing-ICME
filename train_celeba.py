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
from dataset_add_celeba import *

def train_celeba(args):
    same_seeds(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if use_cuda else "cpu")
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

    test_loader = DataLoader(test_dataset, batch_size = args.test_batch_size, shuffle = False)
    domain1_loader = DataLoader(domain1_real_dataset + domain1_print_dataset + domain1_replay_dataset, batch_size = args.batch_size, shuffle = True)
    domain2_loader = DataLoader(domain2_real_dataset + domain2_print_dataset + domain2_replay_dataset, batch_size = args.batch_size, shuffle = True)
    domain3_loader = DataLoader(domain3_real_dataset + domain3_print_dataset + domain3_replay_dataset, batch_size = args.batch_size, shuffle = True)

    domain_a_encoder = torch.nn.DataParallel(torchvision.models.resnet18(pretrained=True), device_ids=[int(args.gpu_id)])
    domain_a_encoder.to(device)
    domain_b_encoder = torch.nn.DataParallel(torchvision.models.resnet18(pretrained=True), device_ids=[int(args.gpu_id)])
    domain_b_encoder.to(device)
    domain_c_encoder = torch.nn.DataParallel(torchvision.models.resnet18(pretrained=True), device_ids=[int(args.gpu_id)])
    domain_c_encoder.to(device)
    shared_content = torch.nn.DataParallel(torchvision.models.resnet18(pretrained=True), device_ids=[int(args.gpu_id)])
    shared_content.to(device)
    shared_spoof = torch.nn.DataParallel(torchvision.models.resnet18(pretrained=True), device_ids=[int(args.gpu_id)])
    shared_spoof.to(device)
    spoof_classify = torch.nn.DataParallel(spoof_classifier_auc(), device_ids=[int(args.gpu_id)])
    spoof_classify.to(device)
    domain_classify = torch.nn.DataParallel(domain_classifier(), device_ids=[int(args.gpu_id)])
    domain_classify.to(device)
    depth_map = torch.nn.DataParallel(depth_decoder(), device_ids=[int(args.gpu_id)])
    depth_map.to(device)
    print("-------------------------------------------------- finish model --------------------------------------------------")

    """## Training"""

    # alpha, beta_depth, beta_faces, gamma = 0.0001, 0.0001, 0.0001, 0.0001  # beta: spoof, gamma: grl
    alpha, beta_depth, beta_faces, gamma = 0.0001, 0.0001, 0.0001, 0.001  # beta: spoof, gamma: grl
    #alpha for spoofing  classify MSE to content and domain
    #gamma for else 
    test_best_auc = 0.0
    test_best_acc = 0.0
    test_best_hter = 0.0
    test_best_epoch = 0

    len_dataloader = min(len(domain1_loader), len(domain2_loader), len(domain3_loader))

    opt_domain_a_encoder = optim.AdamW(domain_a_encoder.parameters(), lr = args.lr)
    opt_domain_b_encoder = optim.AdamW(domain_b_encoder.parameters(), lr = args.lr)
    opt_domain_c_encoder = optim.AdamW(domain_c_encoder.parameters(), lr = args.lr)
    opt_shared_content = optim.AdamW(shared_content.parameters(), lr = args.lr)
    opt_shared_spoof = optim.AdamW(shared_spoof.parameters(), lr = args.lr)
    opt_spoof_classify = optim.AdamW(spoof_classify.parameters(), lr = args.lr)
    opt_domain_classify = optim.AdamW(domain_classify.parameters(), lr = args.lr)
    opt_depth = optim.AdamW(depth_map.parameters(), lr = args.lr)

    opt_domain_a_scheduler = optim.lr_scheduler.MultiStepLR(opt_domain_a_encoder, milestones=[5,25,30, 70, 90,120,150,200,250,300,400], gamma=0.9)
    opt_domain_b_scheduler = optim.lr_scheduler.MultiStepLR(opt_domain_b_encoder, milestones=[5,25,30, 70, 90,120,150,200,250,300,400], gamma=0.9)
    opt_domain_c_scheduler = optim.lr_scheduler.MultiStepLR(opt_domain_c_encoder, milestones=[5,25,30, 70, 90,120,150,200,250,300,400], gamma=0.9)
    opt_shared_content_scheduler = optim.lr_scheduler.MultiStepLR(opt_shared_content, milestones=[5,25,30, 70, 90,120,150,200,250,300,400], gamma=0.9)
    opt_shared_spoof_scheduler = optim.lr_scheduler.MultiStepLR(opt_shared_spoof, milestones=[5,25,30, 70, 90,120,150,200,250,300,400], gamma=0.9)
    opt_spoof_classify_scheduler = optim.lr_scheduler.MultiStepLR(opt_spoof_classify, milestones=[5,25,30, 70, 90,120,150,200,250,300,400], gamma=0.9)
    opt_domain_classify_scheduler = optim.lr_scheduler.MultiStepLR(opt_domain_classify, milestones=[5,25,30, 70, 90,120,150,200,250,300,400], gamma=0.9)
    opt_depth_scheduler = optim.lr_scheduler.MultiStepLR(opt_depth, milestones=[5,25,30, 70, 90,120,150,200,250,300,400], gamma=0.9)

    softmax = nn.Softmax(dim=0)
    class_criterion = nn.CrossEntropyLoss()
    class_criterion_re = MSE()
    mse_loss = MSE()
    # simse_loss = SIMSE()
    # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    print('epoch num = ', args.n_epoch, ', iter num = ', len_dataloader)

    for epoch in range(args.n_epoch):
        domain1_loader = DataLoader(domain1_real_dataset + domain1_print_dataset + domain1_replay_dataset, batch_size = args.batch_size, shuffle = True)
        domain2_loader = DataLoader(domain2_real_dataset + domain2_print_dataset + domain2_replay_dataset, batch_size = args.batch_size, shuffle = True)
        domain3_loader = DataLoader(domain3_real_dataset + domain3_print_dataset + domain3_replay_dataset, batch_size = args.batch_size, shuffle = True)
        print('-------------------------------------------------- epoch = {} --------------------------------------------------'.format(str(epoch))) 
        print('-------------------------------------------------- {} Auc = {} --------------------------------------------------'.format("celeba_spoof", str(test_best_auc)))
        print('-------------------------------------------------- {} Acc = {} --------------------------------------------------'.format("celeba_spoof", str(test_best_acc))) 
        print('-------------------------------------------------- {} Hter = {} --------------------------------------------------'.format("celeba_spoof", str(test_best_hter))) 
        print('-------------------------------------------------- {} @epoch = {} --------------------------------------------------'.format("celeba_spoof", str(test_best_epoch))) 

        e_domain_class_loss = 0.0 
        e_domain_grl_spoof_loss = 0.0 
        e_domain_grl_content_loss = 0.0 
        e_spoof_class_loss = 0.0 
        e_spoof_grl_content_loss = 0.0 
        e_spoof_grl_domain_loss = 0.0 
        e_recon_loss = 0.0 
        e_depth_loss = 0.0
        e_swap_loss = 0.0 
        
        for i, ((d1_data, d1_depth, d1_label), (d2_data, d2_depth, d2_label), (d3_data, d3_depth, d3_label)) in enumerate(zip(domain1_loader, domain2_loader, domain3_loader)):

            domain_a_encoder.train()
            domain_b_encoder.train()
            domain_c_encoder.train()
            shared_content.train()
            shared_spoof.train()
            spoof_classify.train()
            domain_classify.train()
            depth_map.train()

            ###Set iter loss###
            domain_class_loss = 0.0 
            domain_grl_spoof_loss = 0.0 
            domain_grl_content_loss = 0.0 
            spoof_class_loss = 0.0 
            spoof_grl_content_loss = 0.0 
            spoof_grl_domain_loss = 0.0 
            depth_loss = 0.0

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
            mixed_depth = torch.cat([d1_depth, d2_depth, d3_depth], dim = 0).to(device)
            mixed_label = torch.cat([d1_label, d2_label, d3_label], dim = 0).to(device)

            mixed_label_domain = torch.tensor([1/3]).repeat(len_data*3, 3).to(device)
            mixed_label_re = torch.tensor([1/2]).repeat(len_data*3, 2).to(device) # real, print, replay，要讓模型無法分出來

            #設定domain label
            domain_label_true = torch.zeros([len_data*3],dtype=torch.long).to(device)
            domain_label_true[len_data:len_data*2] = 1
            domain_label_true[len_data*2:] = 2
    
            ###Extract feature###
            spoof_feature = shared_spoof(mixed_data)
            content_feature = shared_content(mixed_data)
            domain1_feature = domain_a_encoder(d1_data)
            domain2_feature = domain_b_encoder(d2_data)
            domain3_feature = domain_c_encoder(d3_data)
            domain_feature = torch.cat([domain1_feature, domain2_feature, domain3_feature], dim = 0).to(device)

            # ###Step 1 : 訓練 Domain Classifier(正向訓練)###
            domain_logit = domain_classify(domain_feature)
            domain_class_loss = class_criterion(domain_logit, domain_label_true)
            e_domain_class_loss += domain_class_loss

            loss = domain_class_loss
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
            domain_grl_spoof_loss =  gamma*class_criterion_re(spoof_domain_logit, mixed_label_domain) 
            e_domain_grl_spoof_loss += domain_grl_spoof_loss

            #content部分
            content_domain_logit = softmax(domain_classify(content_feature))
            domain_grl_content_loss = gamma*class_criterion_re(content_domain_logit, mixed_label_domain) 
            e_domain_grl_content_loss += domain_grl_content_loss

            loss = domain_grl_spoof_loss + domain_grl_content_loss
            loss.backward()
            opt_shared_spoof.step()
            opt_shared_content.step()

            opt_shared_spoof.zero_grad() 
            opt_shared_content.zero_grad()

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

            ###Step 4 : 讓Spoof Classify GRL回content和domain###
            content_feature = shared_content(mixed_data) 
            domain1_feature = domain_a_encoder(d1_data)
            domain2_feature = domain_b_encoder(d2_data)
            domain3_feature = domain_b_encoder(d3_data)
            domain_feature = torch.cat([domain1_feature, domain2_feature, domain3_feature], dim = 0).to(device)

            content_logit = softmax(spoof_classify(content_feature, mixed_label_re, False))
            domain_logit = softmax(spoof_classify(domain_feature, mixed_label_re, False))

            spoof_grl_content_loss = gamma*class_criterion_re(content_logit, mixed_label_re)
            spoof_grl_domain_loss = gamma*class_criterion_re(domain_logit, mixed_label_re)
            e_spoof_grl_content_loss += spoof_grl_content_loss
            e_spoof_grl_domain_loss += spoof_grl_domain_loss 

            loss = spoof_grl_content_loss + spoof_grl_domain_loss
            loss.backward()
            opt_shared_content.step()
            opt_domain_a_encoder.step()
            opt_domain_b_encoder.step()
            opt_domain_c_encoder.step()

            opt_shared_content.zero_grad() 
            opt_domain_a_encoder.zero_grad() 
            opt_domain_b_encoder.zero_grad() 
            opt_domain_c_encoder.zero_grad()

            ###Step 5 : 訓練 depth###
            content_feature = shared_content(mixed_data).view(-1, 1000, 1, 1) ###
            depth_recon = depth_map(content_feature)

            err_sim1 = mse_loss(depth_recon, mixed_depth)
            depth_loss = 0.01*err_sim1
            e_depth_loss += depth_loss

            loss = depth_loss
            loss.backward()
            opt_shared_content.step()
            opt_depth.step()
            
            opt_shared_content.zero_grad()
            opt_depth.zero_grad()
            depth_map.eval()

            print("\r {}/{} domain_class_loss:{:.5f}, domain_grl_spoof_loss={:.5f}, domain_grl_content_loss={:.5f}, spoof_class_loss={:.4f}, spoof_grl_content_loss={:.5f}, spoof_grl_domain_loss={:.5f}, depth_loss = {:.5f}".format(
                    i+1, len_dataloader, domain_class_loss.item(), domain_grl_spoof_loss.item(), domain_grl_content_loss.item(), spoof_class_loss.item(), 
                    spoof_grl_content_loss.item(), spoof_grl_domain_loss.item(), depth_loss), end = "")

        opt_domain_a_scheduler.step()
        opt_domain_b_scheduler.step()
        opt_domain_c_scheduler.step()
        opt_shared_content_scheduler.step()
        opt_shared_spoof_scheduler.step()
        opt_spoof_classify_scheduler.step()
        opt_domain_classify_scheduler.step()
        opt_depth_scheduler.step()

        print("{} lr: {}".format(epoch, opt_spoof_classify_scheduler.get_last_lr()[0]))

        print("{}/{} e_domain_class_loss:{:.5f}, e_domain_grl_spoof_loss={:.5f}, e_domain_grl_content_loss={:.5f}, e_spoof_class_loss={:.4f}, e_spoof_grl_content_loss={:.5f}, e_spoof_grl_domain_loss={:.5f}, e_depth_loss = {:.5f}".format(
                i+1, args.n_epoch, e_domain_class_loss.item(), e_domain_grl_spoof_loss.item(), e_domain_grl_content_loss.item(), e_spoof_class_loss.item(), 
                e_spoof_grl_content_loss.item(), e_spoof_grl_domain_loss.item(), e_depth_loss))

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
        # print(pred)
        test_auc = roc_auc_score(ans, pred)
        test_acc = correct/len(test_dataset)
        _, test_hter = HTER(np.array(pred), np.array(ans))

        print('Final {} test auc = {}'.format("celeba_spoof", test_auc))
        print('Final {} test acc = {}'.format("celeba_spoof", test_acc))
        print('Final {} test hter = {}'.format("celeba_spoof", test_hter))

        if test_auc > test_best_auc:
            test_best_auc = test_auc
            test_best_acc = test_acc
            test_best_hter = test_hter
            test_best_epoch = epoch
            torch.save(shared_spoof, shared_spoof_path)
            torch.save(spoof_classify, spoof_classify_path)
            torch.save(shared_content, shared_content_path)
            torch.save(depth_map, depth_map_path)
            torch.save(domain_a_encoder, domain1_encoder_path)
            torch.save(domain_b_encoder, domain2_encoder_path)
            torch.save(domain_c_encoder, domain3_encoder_path)
            torch.save(domain_classify, domain_classify_path)
            print('{}: save model'.format("celeba_spoof"))