# FaceAntiSpoofing-WACV

## Dataset (Please check your number of files is correct)

OULU      | Users | Real access | Print attacks | Video attacks | Total
--------- |:-----:|------------:| -------------:| ------------: |--------
Training	|   20	|    360      |    720	      |     720       |   1800 
Dev       |  15	  |    270	    |     540	      |     540	      |   1350
Test	    |  20	  |    360	    |     720	      |     720	      |   1800 


CASIA     | Users | Real access | Print attacks | Video attacks | Total
--------- |:-----:|------------:| -------------:| ------------: |--------
Training	|   20	|    60       |     120	      |     60        |    240 
Test	    |   30	|    90	      |     180	      |     90	      |    360 


MSU       | Users | Real access | Print attacks | Video attacks | Total
--------- |:-----:|------------:| -------------:| ------------: |--------
Training	|  15	  |    30       |    30	        |     60        |    120
Test	    |  20	  |    40	      |     40	      |     80	      |    160


IDIAP      | Users | Real access | Print attacks | Video attacks | Total
---------  |:-----:|------------:| -------------:| ------------: |--------
Training	 |  ?	   |    60       |    60	       |     240       |    360 
Dev        |  ?	   |    60	     |     60	       |     240	     |    360
Test	     |  ?	   |    80	     |     80	       |     320	     |    480 

SiW        | Users | Real access | Print attacks | Video attacks | Total
---------  |:-----:|------------:| -------------:| ------------: |--------
Training	 |  90	 |    ?        |     ?	       |     ?         |    ? 
Test	     |  75   |    ? 	     |     ?	       |     ?  	     |    ? 

## Usuage 
### 1. Download dataset
**FaceAntiSpoofing-WACV$ download_sample_dataset.py**

You will get

- pr_depth_map_256
  - Oulu_NPU, MSU
  - Replay_Attack
  - CASIA_faceAntisp
  - CelebA_Spoof

### 2. Train model
**FaceAntiSpoofing-WACV$ python3 main.py**

--dataset_path \<the path containing all datasets\> 

--type train_\<training type\> 

--target_domain \<generalized domain\>

### 3. Test model
**FaceAntiSpoofing-WACV$ python3 main.py**

--dataset_path \<the path containing all datasets\>

--type test_auc 

--target_domain \<generalized domain\>

### 4. Explainable AI (Lime)
**FaceAntiSpoofing-WACV$ python3 explainableAI_lime.py**
