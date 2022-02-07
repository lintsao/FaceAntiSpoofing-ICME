import argparse
import sys
import warnings

from train_acc import *
from train_auc import *
from train_binary import *
from train_noContent import *
from train_noContentGRL2Spoof import *
from train_auc_no_depth import *
from train_auc_triplet_ours import *
from train_auc_triplet_ssdg import *
from train_auc_2domain import *
from train_celeba import *
from train_shared_auc import *
from train_shared_nodomain import *
from train_shared_nocontent import *
from test_auc import *

def main(args):
    if args.type == 'train_auc':
        train_auc(args)

    elif args.type == 'train_shared_auc':
        train_shared_auc(args)

    elif args.type == 'train_shared_auc_no_domain':
        train_shared_auc_no_domain(args)

    if args.type == 'train_shared_auc_no_content':
        train_shared_auc_no_content(args)

    elif args.type == 'train_acc':
        train_acc(args)

    elif args.type == 'train_binary':
        train_binary(args)

    elif args.type == 'train_noContent':
        train_noContent(args)

    elif args.type == 'train_noContentGRL2Spoof':
        train_noContentGRL2Spoof(args)

    elif args.type == 'train_noDepth':
        train_noDepth(args)

    elif args.type == 'train_aucTripletOurs':
        train_auc_triplet_ours(args)

    elif args.type == 'train_aucTripletSSDG':
        train_auc_triplet_ssdg(args)

    elif args.type == "train_2domain":
        train_auc_2domain(args)

    elif args.type == 'train_celeba':
        train_celeba(args)

    elif args.type == 'test_auc':
        test_auc(args)

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
    parser.add_argument('--path', type=str, default="./")
    parser.add_argument('--dataset_path', type=str, default="./pr_depth_map_256")
    parser.add_argument('--type', type=str, default='train_auc')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--spoof_encoder', type=str, default="/home/tsaolin/Face_Anti-Spoofing/FaceAntiSpoofing-WACV/model/oulu/224_0/oulu_spoof_encoder.pt")
    parser.add_argument('--spoof_classifier', type=str, default="/home/tsaolin/Face_Anti-Spoofing/FaceAntiSpoofing-WACV/model/oulu/224_0/oulu_spoof_classify.pt")

    # datasets
    parser.add_argument('--target_domain', type=str, default='oulu')
    parser.add_argument('--number_folder', type=str, default='0')
   
    # model
    parser.add_argument('--scratch', type=str, default='True')
    parser.add_argument('--load_folder', type=str, default='0')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.0001)

    # training configs
    parser.add_argument('--img_size', type=int, default=256) 
    parser.add_argument('--depth_size', type=int, default=64) 
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--batch_triplet', type=int, default=4)
    parser.add_argument('--n_epoch', type=int, default=100)

    print(parser.parse_args())
    main(parser.parse_args())