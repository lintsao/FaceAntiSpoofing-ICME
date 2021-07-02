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
from model import *
from dataset import *
from utils import *

def predict(input):
    # input: numpy array, (batches, height, width, channels)
    spoof_encoder.eval()
    spoof_classify.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    # print(input.shape)
    label = torch.LongTensor([0]).repeat(10)
    # 需要先將 input 轉成 pytorch tensor，且符合 pytorch 習慣的 dimension 定義
    # 也就是 (batches, channels, height, width)
    spoof_feature = spoof_encoder(input.cuda())
    output = softmax(spoof_classify(spoof_feature, label.cuda(), False))
    # output = spoof_classify(spoof_feature, label.cuda(), False)
    # print(output.detach().cpu().numpy())

    return output.detach().cpu().numpy()

def segmentation(input):
    # 利用 skimage 提供的 segmentation 將圖片分成 50 塊
    return slic(input, n_segments=15, compactness=10, sigma=1, start_label=1)

same_seeds(0)

"""## Argument parsing"""

args = {
      'target': 'oulu',
      'type': 'replay_fixed',
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
path = './dataset'
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
print("-------------------------------------------------- finish dataset --------------------------------------------------")

"""## Model definition and checkpoint loading"""

if args.target == 'msu':    
    spoof_encoder = torch.load(args.spoof_encoder_msu)
    spoof_classify = torch.load(args.spoof_classify_msu)
    if args.type == 'real':
        test_dataset = msu_test_real_dataset
    if args.type == 'print':
        test_dataset = msu_test_print_dataset
    if args.type == 'replay':
        test_dataset = msu_test_replay_dataset

elif args.target == 'idiap':    
    spoof_encoder = torch.load(args.spoof_encoder_idiap)
    spoof_classify = torch.load(args.spoof_classify_idiap)
    if args.type == 'real':
        test_dataset = idiap_test_real_dataset
    if args.type == 'print_fixed':
        test_dataset = idiap_test_print_fixed_dataset
    if args.type == 'replay_fixed':
        test_dataset = idiap_test_replay_fixed_dataset
    if args.type == 'print_hand':
        test_dataset = idiap_test_print_hand_dataset
    if args.type == 'replay_hand':
        test_dataset = idiap_test_replay_hand_dataset

elif args.target == 'oulu':    
    spoof_encoder = torch.load(args.spoof_encoder_oulu)
    spoof_classify = torch.load(args.spoof_classify_oulu)
    test_dataset = oulu_test_dataset

elif args.target == 'casia':    
    spoof_encoder = torch.load(args.spoof_encoder_casia)
    spoof_classify = torch.load(args.spoof_classify_casia)
    test_dataset = casia_test_dataset
print("-------------------------------------------------- finish model --------------------------------------------------")

"""## Lime
Lime 的部分因為有現成的套件可以使用，因此下方直接 demo 如何使用該套件
其實非常的簡單，只需要 implement 兩個 function 即可
"""

softmax = nn.Softmax(dim=1)

# img_indices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
# img_indices = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
# img_indices = [0, 1, 2, 3, 4]
img_indices = list(range(len(test_dataset)))
images, labels = test_dataset.getbatch('test', img_indices)

spoof_encoder.eval()
spoof_classify.eval()
img_indices_new = []
real_num = 0 
print_num = 0
replay_num = 0
with torch.no_grad():
    for batch_idx, (image, label) in enumerate(zip(images, labels)):
        image = image.unsqueeze(0).cuda()
        label = label.unsqueeze(0).cuda()
        spoof_feature = spoof_encoder(image)
        output = softmax(spoof_classify(spoof_feature, label, False))
        for j in range(len(output)):
            print(label[j].item(), torch.argmax(output[j], dim=0).item())
            if label[j].item() == torch.argmax(output[j], dim=0).item():
                if label[j].item() == 0 and real_num < 4:
                    img_indices_new.append(img_indices[batch_idx])
                    real_num += 1
                if label[j].item() == 1 and print_num < 3:
                    img_indices_new.append(img_indices[batch_idx])
                    print_num += 1
                if label[j].item() == 2 and replay_num < 3:
                    img_indices_new.append(img_indices[batch_idx])
                    replay_num += 1
                # print(batch_idx, label[j].item(), torch.argmax(output[j], dim=0).item())

# 讓實驗 reproducible
limit_num = 10
images, labels = test_dataset.getbatch('test', img_indices_new)
fig, axs = plt.subplots(2, limit_num, figsize=(15, 8))
for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    print(idx, label)
    x = image.astype(np.double)
    # lime 這個套件要吃 numpy array

    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation, top_labels=3)
    # 基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
    # classifier_fn 定義圖片如何經過 model 得到 prediction
    # segmentation_fn 定義如何把圖片做 segmentation

    lime_img, mask = explaination.get_image_and_mask(
                                label=label.item(),
                                positive_only=False,
                                # negative_only=True,
                                hide_rest=False,
                                num_features=3,
                                min_weight=0.0001
                            )
    # 把 explainer 解釋的結果轉成圖片

    if label == 0:
        axs[0][idx].set_title('real')
    elif label == 1:
        axs[0][idx].set_title('print')
    elif label == 2:
        axs[0][idx].set_title('replay')

    axs[0][idx].imshow(image)
    axs[1][idx].imshow(lime_img)
    if idx+1 >= limit_num:
        break

plt.savefig('lime_{}.png'.format(args.target))
