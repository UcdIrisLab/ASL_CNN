# ASL_CNN


The process is divided into two steps
 - Preprocessing
 - Training

File structure is as following:

/PATH/Data_1
- Data_1_ATT.nii
- Data_1_CBF.nii
- brain_mask.nii
- PWI_timing.nii (inlcuding PWI image and a timing information.)

For preprocessing,  
 python pre_proc.py --path /PATH/ --subjects Data_1 Data_2 Data_3 Data_4 --base --conv3d
 
For training,
 python MRI-DK.py --savename clean_UP --model conv3d_staged --PWI 7 --train_subjects Data_1 Data_2 Data_3 --epoch 200 --test_subject Data_4 --train
 - --PWI requires number of PLD+1 which compromise PLDs and a delay information

Using pre-trained weights.
Move the pre-traind weight files to ./weights/
python MRI-DK.py --savename weight_name (i.e. 6PLDs) --model conv3d_staged --PWI 7 --train_subjects Data_1 Data_2 Data_3 --epoch 200 --test_subject Data_4 

