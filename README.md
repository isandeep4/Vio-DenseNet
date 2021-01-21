# ViolenceDetector (Temporal Segment Network)
This repository holds the codes and models for the papers.
https://arxiv.org/abs/1705.02953

Temporal Segment Networks for Action Recognition in Videos, Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool, TPAMI, 2018.

Data Preprocessing:
  1)Data_generator - Creates batch size and samples n_sequence frame from video file.Used rgbDiff function to calculate optical flow between two consequtive sample frames.
  2)Data_helper - read text file and return a dictionary
  3)Data_Preprocessing - write train and validation text file for input videos data with labeled set.
  
  
 Modelling: 
    - Used keras sequential model and layers like LSTM, CuDNNLSTM and includes pretrained model of MobileNetV2 trained on imagenet.
    -  run webcam.py to see the label output on the test video using opencv.
    
 
