# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace
import cv2

from model import *
from dataset import *
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img,(256, 256))
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.0,],[1.0,])
    ])
    img_input = img_transform(img, transform)
    return img_input


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))


def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (32, 32))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


if __name__ == '__main__':

    BASE_DIR = "./"
    path = './dataset'
    path_img = "./Oulu_NPU/Test_files/crop_frame/crop_face/cropped_1_1_36_1.jpg"
    output_dir = BASE_DIR

    """## Argument parsing"""

    args = {
        'spoof_encoder_msu': './model/msu/2/msu_spoof_encoder.pt',
        'spoof_classify_msu': './model/msu/2/msu_spoof_classify.pt',
        'spoof_encoder_idiap': './model/idiap/2/idiap_spoof_encoder.pt',
        'spoof_classify_idiap': './model/idiap/2/idiap_spoof_classify.pt',
        'spoof_encoder_oulu': './model/oulu/2/oulu_spoof_encoder.pt',
        'spoof_classify_oulu': './model/oulu/2/oulu_spoof_classify.pt',
        'spoof_encoder_casia': './model/casia/2/casia_spoof_encoder.pt',
        'spoof_classify_casia': './model/casia/2/casia_spoof_classify.pt',
    }
    args = argparse.Namespace(**args)

    """## Model definition and checkpoint loading"""
    target = 'oulu'

    if target == 'msu':    
        spoof_encoder = torch.load(args.spoof_encoder_msu)
        spoof_classify = torch.load(args.spoof_classify_msu)
    elif target == 'idiap':    
        spoof_encoder = torch.load(args.spoof_encoder_idiap)
        spoof_classify = torch.load(args.spoof_classify_idiap)
    elif target == 'oulu':    
        spoof_encoder = torch.load(args.spoof_encoder_oulu)
        spoof_classify = torch.load(args.spoof_classify_oulu)
    elif target == 'casia':    
        spoof_encoder = torch.load(args.spoof_encoder_casia)
        spoof_classify = torch.load(args.spoof_classify_casia)

    """## Dataset definition and creation"""

    # 助教 training 時定義的 dataset
    # 因為 training 的時候助教有使用底下那些 transforms，所以 testing 時也要讓 test data 使用同樣的 transform
    # dataset 這部分的 code 基本上不應該出現在你的作業裡，你應該使用自己當初 train HW3 時的 preprocessing
    img_size=256
    depth_size=64

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
    msu_test_real_dataset = MSU_dataset(os.path.join(path, 'MSU/dataset/scene01/real/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "real")
    msu_test_print_dataset = MSU_dataset(os.path.join(path, 'MSU/dataset/scene01/attack/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "print")
    msu_test_replay_dataset = MSU_dataset(os.path.join(path, 'MSU/dataset/scene01/attack/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "replay")

    idiap_test_real_dataset = Idiap_dataset(os.path.join(path, 'Replay_Attack/replayattack-test/test/real/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "real")
    idiap_test_print_fixed_dataset = Idiap_dataset(os.path.join(path, 'Replay_Attack/replayattack-test/test/attack/fixed/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "print")
    idiap_test_replay_fixed_dataset = Idiap_dataset(os.path.join(path, 'Replay_Attack/replayattack-test/test/attack/fixed/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "replay")
    idiap_test_print_hand_dataset = Idiap_dataset(os.path.join(path, 'Replay_Attack/replayattack-test/test/attack/hand/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "print")
    idiap_test_replay_hand_dataset = Idiap_dataset(os.path.join(path, 'Replay_Attack/replayattack-test/test/attack/hand/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth, attack = "replay")

    oulu_test_dataset = Oulu_dataset(os.path.join(path, 'Oulu_NPU/Test_files/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth)

    casia_test_dataset = Casia_dataset(os.path.join(path, 'CASIA_faceAntisp/test_release/crop_frame/crop_face'), 'test', transform = transform, transform_depth = transform_depth)

    classes = ('real', 'print', 'replay')
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img)
    net = spoof_encoder

    # 注册hook
    net.conv2.register_forward_hook(farward_hook)
    net.conv2.register_backward_hook(backward_hook)

    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output)
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (32, 32))) / 255
    show_cam_on_image(img_show, cam, output_dir)