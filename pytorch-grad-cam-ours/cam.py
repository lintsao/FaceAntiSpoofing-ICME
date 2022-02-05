import argparse
import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import os
import sys

from model import *

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default="/home/tsaolin/Face_Anti-Spoofing/pr_depth_map_256/CelebA_Spoof/spoof/",
                        help='Input image path')
    parser.add_argument('--folder_path', type=str, default='/home/tsaolin/Face_Anti-Spoofing/pr_depth_map_256/CelebA_Spoof/real/',
                        help='Input image folder path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--target_category', type=int, default=None)
    parser.add_argument('--mode', type=str, default='proposed')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        device = "cuda"
        print('Using GPU for acceleration')
    else:
        device = "cpu"
        print('Using CPU for computation')

    return args, device

class proposedModel(nn.Module):
    def __init__(self, liveness_encoder, liveness_classifier):
        super(proposedModel, self).__init__()
        self.liveness_encoder = liveness_encoder
        self.liveness_classifier = liveness_classifier

    def forward(self, x):
        x = self.liveness_encoder(x)
        out = self.liveness_classifier(x, torch.tensor([[1]]), False)
        # print(out.shape)
        return out

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Feature_Generator_MADDG(nn.Module):
    def __init__(self):
        super(Feature_Generator_MADDG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(128)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv1_3 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(196)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.conv1_4 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(128)
        self.relu1_4 = nn.ReLU(inplace=True)
        self.maxpool1_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_5 = nn.BatchNorm2d(128)
        self.relu1_5 = nn.ReLU(inplace=True)
        self.conv1_6 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_6 =  nn.BatchNorm2d(196)
        self.relu1_6 = nn.ReLU(inplace=True)
        self.conv1_7 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_7 = nn.BatchNorm2d(128)
        self.relu1_7 = nn.ReLU(inplace=True)
        self.maxpool1_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1_8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_8 = nn.BatchNorm2d(128)
        self.relu1_8 = nn.ReLU(inplace=True)
        self.conv1_9 = nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_9 = nn.BatchNorm2d(196)
        self.relu1_9 = nn.ReLU(inplace=True)
        self.conv1_10 = nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_10 = nn.BatchNorm2d(128)
        self.relu1_10 = nn.ReLU(inplace=True)
        self.maxpool1_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.bn1_1(out)
        out = self.relu1_1(out)
        out = self.conv1_2(out)
        out = self.bn1_2(out)
        out = self.relu1_2(out)
        out = self.conv1_3(out)
        out = self.bn1_3(out)
        out = self.relu1_3(out)
        out = self.conv1_4(out)
        out = self.bn1_4(out)
        out = self.relu1_4(out)
        pool_out1 = self.maxpool1_1(out)

        out = self.conv1_5(pool_out1)
        out = self.bn1_5(out)
        out = self.relu1_5(out)
        out = self.conv1_6(out)
        out = self.bn1_6(out)
        out = self.relu1_6(out)
        out = self.conv1_7(out)
        out = self.bn1_7(out)
        out = self.relu1_7(out)
        pool_out2 = self.maxpool1_2(out)

        out = self.conv1_8(pool_out2)
        out = self.bn1_8(out)
        out = self.relu1_8(out)
        out = self.conv1_9(out)
        out = self.bn1_9(out)
        out = self.relu1_9(out)
        out = self.conv1_10(out)
        out = self.bn1_10(out)
        out = self.relu1_10(out)
        pool_out3 = self.maxpool1_3(out)
        return pool_out3

class Feature_Embedder_MADDG(nn.Module):
    def __init__(self):
        super(Feature_Embedder_MADDG, self).__init__()
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottleneck_layer_1 = nn.Sequential(
            self.conv3_1,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.pool2_1,
            self.conv3_2,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self.pool2_2,
            self.conv3_3,
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self, input, norm_flag):
        feature = self.bottleneck_layer_1(input)
        feature = self.avg_pool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)
        if (norm_flag):
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
        return feature

from torchvision.models.resnet import ResNet, BasicBlock
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # change your path
    # model_path = '/home/jiayunpei/SSDG_github/pretrained_model/resnet18-5c106cde.pth'
    model_path = "drive/My Drive/senior_2/Face_AntiSpoofing/related_work_experiment/resnet18-5c106cde.pth"
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("loading model: ", model_path)
    # print(model)
    return model

class Feature_Generator_ResNet18(nn.Module):
    def __init__(self):
        super(Feature_Generator_ResNet18, self).__init__()
        model_resnet = resnet18(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        return feature

class Feature_Embedder_ResNet18(nn.Module):
    def __init__(self):
        super(Feature_Embedder_ResNet18, self).__init__()
        model_resnet = resnet18(pretrained=False)
        self.layer4 = model_resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.bottleneck_layer_fc = nn.Linear(512, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, input, norm_flag):
        feature = self.layer4(input)
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck_layer(feature)
        if (norm_flag):
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
        return feature

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(512, 3)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL()

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out

class DG_model(nn.Module):
    def __init__(self, model):
        super(DG_model, self).__init__()
        if(model == 'resnet18'):
            self.backbone = Feature_Generator_ResNet18()
            self.embedder = Feature_Embedder_ResNet18()
        elif(model == 'maddg'):
            self.backbone = Feature_Generator_MADDG()
            self.embedder = Feature_Embedder_MADDG()
        else:
            print('Wrong Name!')
        self.classifier = Classifier()

    def forward(self, input, norm_flag):
        feature = self.backbone(input)
        feature = self.embedder(feature, norm_flag)
        classifier_out = self.classifier(feature, norm_flag)
        return classifier_out, feature

if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args, device = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    if args.mode == 'proposed':
        Cb_liveness_encoder = torch.load("./Cb_liveness_encoder.pt", map_location=device)
        print("finish liveness_encoder")
        Cb_liveness_classifier = torch.load("./Cb_liveness_classifier.pt", map_location=device)
        print("finish liveness_classifier")
        model = proposedModel(Cb_liveness_encoder, Cb_liveness_classifier)
        print(model)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        target_layer = model.liveness_encoder.layer4[-1]
        print(target_layer)

    elif args.mode == 'ssdg':
        model = torch.load("/home/tsaolin/Face_Anti-Spoofing/SSDG-CVPR2020/ssdg.pt", map_location=device)
        print(model)
        print("model")
        target_layer = model.embedder.layer4[-1]
        print(target_layer)

    else:
        print("no such mode")
        sys.exit()

    print("="*60)
    images = sorted(os.listdir(args.folder_path))
    print("total image: {}".format(len(images)))
    print("="*60)

    for img_path in sorted(images):
        print(img_path)
        rgb_img = cv2.imread(os.path.join(args.folder_path, img_path), 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        if args.mode == 'proposed':
            input_tensor = preprocess_image(rgb_img, mean=[0.0,], std=[1.0,])
        elif args.mode == 'ssdg':
            input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        print("finish image")
        cam = methods[args.method](model=model,
                                target_layer=target_layer,
                                use_cuda=args.use_cuda)

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        
        # If None, returns the map for the highest scoring category. Otherwise, targets the requested category.
        target_category = 1 # 0: real, 1: fake
        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        # gb = gb_model(input_tensor, target_category=target_category)

        # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        # cam_gb = deprocess_image(cam_mask * gb)
        # gb = deprocess_image(gb)

        name = img_path.split('.')[0]

        cv2.imwrite('Cb_ssdg_real/{}_{}_{}_cam.jpg'.format(name, args.mode, args.method), cam_image)
        # cv2.imwrite(f'{args.method}_gb.jpg', gb)
        # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)

        print("finish", name)
        print("="*60)
