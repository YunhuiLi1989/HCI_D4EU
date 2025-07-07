# HCI_D4EU
Code for "Dual-Domain Joint Enhanced Unfolding with Efficient Attention for Snapshot Hyperspectral Compressive Imaging Reconstruction"

The manuscript has been submitted to IEEE Transactions on Circuits and Systems for Video Technology.

# Requirements

scipy~=1.11.1
numpy~=1.25.2
torchvision~=0.15.2
einops~=0.6.1
pandas~=2.1.0  

# Prepare Dataset
	We applied the same dataset as the following literature in our code: 
	      "Yuanhao Cai, Jing Lin, etal. Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction. In CVPR 2022".
	The download link for the dataset shared in the above literature is as follows: 
	cave_1024_28:       'https://pan.baidu.com/share/init?surl=X_uXxgyO-mslnCTn4ioyNQ' (code: fo0q)
	TSA_simu_data:     'https://pan.baidu.com/share/init?surl=LI9tMaSprtxT8PiAG1oETA' (code: efu8)
	CAVE_512_28:        'https://pan.baidu.com/share/init?surl=ue26weBAbn61a7hyT9CDkg' (code: ixoe)
	KAIST_CVPR2021: 'https://pan.baidu.com/share/init?surl=LfPqGe0R_tuQjCXC_fALZA' (code: 5mmn)
        TSA_real_data:       'https://pan.baidu.com/share/init?surl=RoOb1CKsUPFu0r01tRi5Bg' (code: eaqe)

	For simulation:
	Download the training dataset from 'https://pan.baidu.com/share/init?surl=X_uXxgyO-mslnCTn4ioyNQ' (code: fo0q) and place them to 'HCI_D4EU\Simulation\CAVE_1024_28'.
	Download the testing dataset from 'https://pan.baidu.com/share/init?surl=LI9tMaSprtxT8PiAG1oETA' (code: efu8) and place them to 'HCI_D4EU\Simulation\Test_data'.

	For real scene:
	Download the training dataset from 'https://pan.baidu.com/share/init?surl=ue26weBAbn61a7hyT9CDkg' (code: ixoe) and place them to 'HCI_D4EU\Real\dataset\CAVE_512_28'.
	Download the training dataset from 'https://pan.baidu.com/share/init?surl=LfPqGe0R_tuQjCXC_fALZA' (code: 5mmn) and place them to  'HCI_D4EU\Real\dataset\KAIST_CVPR2021'.
	Download the testing dataset from 'https://pan.baidu.com/share/init?surl=RoOb1CKsUPFu0r01tRi5Bg' (code: eaqe) and place them to 'HCI_D4EU\Real\dataset\TSA_real_data'.

# Simulation
	Training
	Run "Train.py".
	
	Testing
	Download the pre-trained model from 'https://pan.baidu.com/s/15qH48MCruOcml34T4iY8CQ' (code: D4EU) and place them to 'HCI_D4EU\Simulation\Checkpoint'.
	Run "Test.py".

# Real Scene
	Training
	Run "train.py".
	
	Testing
	Download the pre-trained model from 'https://pan.baidu.com/s/1PT2QwzXAc4tAMPl6n40SNA' (code: D4EU) and place them to 'HCI_D4EU\Real\train_code\output\model'.
	Run "test.py".





