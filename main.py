from train import *
import sys

if __name__ == '__main__':
    target_name = sys.argv[1]
    component_name = sys.argv[2]
    train(path='./', dataset_path='./dataset', target_domain=target_name, number_folder=target_name+'_'+component_name, img_size=256, depth_size=64, batch_size=24, batch_triplet=4, lr=0.0002, n_epoch=100)

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