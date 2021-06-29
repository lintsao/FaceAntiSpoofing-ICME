from train import *

if __name__ == '__main__':
    train(path='./', dataset_path='./dataset', target_domain='oulu', number_folder='2', img_size=256, depth_size=64, batch_size=8, batch_triplet=4, lr=0.0003, n_epoch=200)

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