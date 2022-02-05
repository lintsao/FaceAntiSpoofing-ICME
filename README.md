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
### 1. unzip dataset
**unzip dataset.zip**

You will get

- training folder: pr_depth_map
- testing folder: Oulu_NPU, MSU, Replay_Attack, CASIA_faceAntisp

### 2. Train model
**FaceAntiSpoofing-WACV$ python3 main.py**

### 3. Test model
**FaceAntiSpoofing-WACV$ python3 test.py**

### 4. Explainable AI (Lime)
**FaceAntiSpoofing-WACV$ python3 explainableAI_lime.py**

