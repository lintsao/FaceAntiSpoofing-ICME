from train import *
import argparse
# from train_last import *

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
    parser.add_argument('--dataset_path', type=str, default='./dataset')

    # datasets 
    parser.add_argument('--target_domain', type=str, default='oulu')
    parser.add_argument('--number_folder', type=str, default='0')
   
    # model

    # optimizer
    parser.add_argument('--lr', type=float, default=0.0003)

    # # # # training configs
    parser.add_argument('--img_size', type=int, default=256) 
    parser.add_argument('--depth_size', type=int, default=64) 
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--batch_triplet', type=int, default=4) 
    parser.add_argument('--n_epoch', type=int, default=100)

    print(parser.parse_args())
    train(parser.parse_args())
