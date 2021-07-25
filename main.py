import argparse
import sys
import warnings

from train_acc import *
from train_auc import *
from train_binary import *
from train_noContent import *
from train_noContentGRL2Spoof import *
from train_auc_no_depth import *
def main(args):
    if args.type == 'auc':
        train_auc(args)
    elif args.type == 'acc':
        train_acc(args)
    elif args.type == 'binary':
        train_binary(args)
    elif args.type == 'noContent':
        train_noContent(args)
    elif args.type == 'noContentGRL2Spoof':
        train_noContentGRL2Spoof(args)
    elif args.type == 'noDepth':
        train_noDepth(args)
    else:
        warnings.warn("Please check your training type")
        sys.exit()

if __name__ == '__main__':
    # path: model will be saved to path/model/target_domain/number_folder/model.pt   
    # dataset_path: folder of training and testing file (includes pr_depth_map)  
    # target_domain 
    # number_folder  
    # img_size: resize img
    # depth_size: resize depth img
    # batch_size 
    # batch_triplet 
    # lr
    # n_epoch

    parser = argparse.ArgumentParser(description="FRANK")

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--dataset_path', type=str, default='./dataset_224')
    parser.add_argument('--type', type=str, default='auc')
    parser.add_argument('--gpu_id', type=str, default='0')

    # datasets 
    parser.add_argument('--target_domain', type=str, default='oulu')
    parser.add_argument('--number_folder', type=str, default='0')
   
    # model

    # optimizer
    parser.add_argument('--lr', type=float, default=0.0001)

    # training configs
    parser.add_argument('--img_size', type=int, default=224) 
    parser.add_argument('--depth_size', type=int, default=64) 
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--batch_triplet', type=int, default=4) 
    parser.add_argument('--n_epoch', type=int, default=100)

    print(parser.parse_args())
    main(parser.parse_args())
