# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import sys
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Function
import argparse
import csv
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import time
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from utils import *
from model import *
from loss import *
from dataset_add_celeba import *

def tsne(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("finish initialization, device: {}".format(device))

    test_real_dataset, test_spoof_dataset, domain1_real_dataset, domain1_print_dataset, domain1_replay_dataset, \
    domain2_real_dataset, domain2_print_dataset, domain2_replay_dataset, domain3_real_dataset, \
    domain3_print_dataset, domain3_replay_dataset = choose_dataset(args.dataset_path, args.target_domain, args.img_size, args.depth_size)

    print("-------------------------------------------------- finish dataset --------------------------------------------------")

    print("test_dataset:{}".format(len(test_real_dataset + test_spoof_dataset)))
    print("domain1_dataset:{}".format(len(domain1_real_dataset + domain1_print_dataset + domain1_replay_dataset)))
    print("domain2_dataset:{}".format(len(domain2_real_dataset + domain2_print_dataset + domain2_replay_dataset)))
    print("domain3_dataset:{}".format(len(domain3_real_dataset + domain3_print_dataset + domain3_replay_dataset)))

    test_real_loader = DataLoader(test_real_dataset, batch_size = args.batch_size, shuffle = False)
    test_fake_loader = DataLoader(test_spoof_dataset, batch_size = args.batch_size, shuffle = False)
    domain1_real_loader = DataLoader(domain1_real_dataset, batch_size = args.batch_size, shuffle = False)
    domain1_print_loader = DataLoader(domain1_print_dataset, batch_size = args.batch_size, shuffle = False)
    domain1_replay_loader = DataLoader(domain1_replay_dataset, batch_size = args.batch_size, shuffle = False)
    domain2_real_loader = DataLoader(domain2_real_dataset, batch_size = args.batch_size, shuffle = False)
    domain2_print_loader = DataLoader(domain2_print_dataset, batch_size = args.batch_size, shuffle = False)
    domain2_replay_loader = DataLoader(domain2_replay_dataset, batch_size = args.batch_size, shuffle = False)
    domain3_real_loader = DataLoader(domain3_real_dataset, batch_size = args.batch_size, shuffle = False)
    domain3_print_loader = DataLoader(domain3_print_dataset, batch_size = args.batch_size, shuffle = False)
    domain3_replay_loader = DataLoader(domain3_replay_dataset, batch_size = args.batch_size, shuffle = False)
    # Ethen

    # share_feature = torch.load()
    shared_spoof = torch.load("/home/tsaolin/Face_Anti-Spoofing/FaceAntiSpoofing-WACV/idiap_spoof_encoder.pt", map_location=device)
    # share_feature.eval()
    shared_spoof.eval()

    features = torch.tensor([]).to(device)
    attacks = torch.tensor([]).to(device)
    domains = torch.tensor([]).to(device)

    with torch.no_grad():
        sample_num = 40
        # MSU real: 'blue'
        for batch_idx, data in enumerate(domain1_real_loader):
            print("\r", batch_idx, '/', len(domain1_real_loader), end = "")
            im, _, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)
            result = shared_spoof(im)
            features = torch.cat((features, result), 0)
            attacks = torch.cat((attacks, label), 0)
            domains = torch.cat((domains, torch.Tensor([0]).repeat(len(im)).to(device)), 0)
            if batch_idx >= sample_num:
                break
        # MSU print: 'blue'
        for batch_idx, data in enumerate(domain1_print_loader):
            print("\r", batch_idx, '/', len(domain1_print_loader), end = "")
            im, _, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)
            result = shared_spoof(im)
            features = torch.cat((features, result), 0)
            attacks = torch.cat((attacks, label), 0)
            domains = torch.cat((domains, torch.Tensor([0]).repeat(len(im)).to(device)), 0)
            if batch_idx >= sample_num:
                break
        # MSU replay: 'blue'
        for batch_idx, data in enumerate(domain1_replay_loader):
            print("\r", batch_idx, '/', len(domain1_replay_loader), end = "")
            im, _, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)
            result = shared_spoof(im)
            features = torch.cat((features, result), 0)
            attacks = torch.cat((attacks, label), 0)
            domains = torch.cat((domains, torch.Tensor([0]).repeat(len(im)).to(device)), 0)
            if batch_idx >= sample_num:
                break
        ###########################
        # IDIAP real: 'orange'
        for batch_idx, data in enumerate(domain2_real_loader):
            print("\r", batch_idx, '/', len(domain2_real_loader), end = "")
            im, _, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)
            result = shared_spoof(im)
            features = torch.cat((features, result), 0)
            attacks = torch.cat((attacks, label), 0)
            domains = torch.cat((domains, torch.Tensor([1]).repeat(len(im)).to(device)), 0)
            if batch_idx >= sample_num:
                break
        # IDIAP print: 'orange'
        for batch_idx, data in enumerate(domain2_print_loader):
            print("\r", batch_idx, '/', len(domain2_print_loader), end = "")
            im, _, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)
            result = shared_spoof(im)
            features = torch.cat((features, result), 0)
            attacks = torch.cat((attacks, label), 0)
            domains = torch.cat((domains, torch.Tensor([1]).repeat(len(im)).to(device)), 0)
            if batch_idx >= sample_num:
                break
        # IDIAP replay: 'orange'
        for batch_idx, data in enumerate(domain2_replay_loader):
            print("\r", batch_idx, '/', len(domain2_replay_loader), end = "")
            im, _, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)
            result = shared_spoof(im)
            features = torch.cat((features, result), 0)
            attacks = torch.cat((attacks, label), 0)
            domains = torch.cat((domains, torch.Tensor([1]).repeat(len(im)).to(device)), 0)
            if batch_idx >= sample_num:
                break
        ###########################
        # OULU real: 'green'
        for batch_idx, data in enumerate(domain3_real_loader):
            print("\r", batch_idx, '/', len(domain3_real_loader), end = "")
            im, _, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)
            result = shared_spoof(im)
            features = torch.cat((features, result), 0)
            attacks = torch.cat((attacks, label), 0)
            domains = torch.cat((domains, torch.Tensor([2]).repeat(len(im)).to(device)), 0)
            if batch_idx >= sample_num:
                break
        # OULU print: 'green'
        for batch_idx, data in enumerate(domain3_print_loader):
            print("\r", batch_idx, '/', len(domain3_print_loader), end = "")
            im, _, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)
            result = shared_spoof(im)
            features = torch.cat((features, result), 0)
            attacks = torch.cat((attacks, label), 0)
            domains = torch.cat((domains, torch.Tensor([2]).repeat(len(im)).to(device)), 0)
            if batch_idx >= sample_num:
                break
        # OULU replay: 'green'
        for batch_idx, data in enumerate(domain3_replay_loader):
            print("\r", batch_idx, '/', len(domain3_replay_loader), end = "")
            im, _, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)
            result = shared_spoof(im)
            features = torch.cat((features, result), 0)
            attacks = torch.cat((attacks, label), 0)
            domains = torch.cat((domains, torch.Tensor([2]).repeat(len(im)).to(device)), 0)
            if batch_idx >= sample_num:
                break
        ###########################
        # CELEBA-SPOOF real: 'red'
        # celebaSpoof_ok = [3, 35, 24, 68, 54, 76, 95, 39, 69, 79, 87, 86, 50, 152, 189, 183, 111, 135]
        for batch_idx, data in enumerate(test_real_loader):
            # if batch_idx+2 not in celebaSpoof_ok:
            #     continue
            print("\r", batch_idx, '/', len(test_real_loader), end = "")
            im, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)
            result = shared_spoof(im)
            features = torch.cat((features, result), 0)
            attacks = torch.cat((attacks, label), 0)
            # attacks = torch.cat((attacks, torch.tensor([batch_idx+2]).to(device)), 0)
            domains = torch.cat((domains, torch.Tensor([3]).repeat(len(im)).to(device)), 0)
            if batch_idx >= 40:
                break
        # CELEBA-SPOOF fake: 'red'
        for batch_idx, data in enumerate(test_fake_loader):
            print("\r", batch_idx, '/', len(test_fake_loader), end = "")
            im, label = data
            im, label = im.to(device), label.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)
            result = shared_spoof(im)
            features = torch.cat((features, result), 0)
            attacks = torch.cat((attacks, label), 0)
            domains = torch.cat((domains, torch.Tensor([3]).repeat(len(im)).to(device)), 0)
            if batch_idx >= 40:
                break
        
        features = features.data.cpu().numpy()
        attacks = attacks.data.cpu().numpy()
        domains = domains.data.cpu().numpy()

    # tsne
    features_embedded = TSNE(n_components=2, perplexity=10.0, early_exaggeration=2.0, learning_rate=200.0, random_state=0, n_iter=10000).fit_transform(features)

    plt.figure(figsize=(16, 16))
    # plt.title("O&M&I to CelebA-Spoof", fontsize=30) # title
    attack_map = ['O', 'X'] + list(range(200))
    color_map = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    X_tsne = features_embedded    

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # normalize
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], attack_map[int(attacks[i])], color=color_map[int(domains[i])], fontdict={'weight': 'bold', 'size': 13})
    plt.xticks([])
    plt.yticks([])

    # for plt.legend
    legend_dict = { 'src1_real' : 'blue', 'src1_fake': 'blue', 'src2_real' : 'orange', 'src2_fake': 'orange', \
        'src3_real' : 'green', 'src3_fake': 'green', 'tgt_real' : 'red', 'tgt_fake': 'red'}
    patchList = []
    for key in legend_dict:
        if 'real' in key:
            data_key = Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor='w', markeredgecolor=legend_dict[key], markersize=10, markeredgewidth=2.0)
        else:
            data_key = Line2D([0], [0], marker='x', color='w', label=key, markerfacecolor='w', markeredgecolor=legend_dict[key], markersize=10, markeredgewidth=2.0)
        patchList.append(data_key)
    # loc = ['upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center]
    plt.legend(handles=patchList, loc = 'lower left', prop = {'size':20})

    plt.savefig('./plot.png') 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="FRANK")

    parser.add_argument('--dataset_path', type=str, default='../pr_depth_map_256')
    parser.add_argument('--gpu_id', type=str, default='0')

    # datasets 
    parser.add_argument('--target_domain', type=str, default='casia')
    parser.add_argument('--number_folder', type=str, default='0')

    # configs
    parser.add_argument('--img_size', type=int, default=256) 
    parser.add_argument('--depth_size', type=int, default=64) 
    parser.add_argument('--batch_size', type=int, default=1)

    print(parser.parse_args())
    tsne(parser.parse_args())
