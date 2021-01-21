# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 09:45:24 2021

@author: isand
"""
import cv2
import numpy as np
from model import Create_pretrained_model
import time
from data_helper import calculateRGBdiff

dim = (224,224)
n_sequence = 8
n_channels = 3
n_output = 2
batch_size = 2
weights_path = './save_weight/weight-39-0.97-0.94.hdf5' 

model = Create_pretrained_model(dim, n_sequence, n_channels, n_output)
model.load_weights(weights_path)
frame_window = np.empty((0, *dim, n_channels))
RUN_STATE = 0
WAIT_STATE = 1
SET_NEW_ACTION_STATE = 2
state = RUN_STATE # 
previous_action = -1 # no action
text_show = 'no action'

class_text = ['violent', 'non-violent']
cap = cv2.VideoCapture('newfi83.avi')
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,dim)
    frame_new = frame/255.0
    frame_new_rs = np.reshape(frame_new,(1,*frame_new.shape))
    frame_window = np.append(frame_window,frame_new_rs,axis=0)
    
    if frame_window.shape[0] >= n_sequence:
            frame_window_dif = calculateRGBdiff(frame_window.copy())
            frame_window_new = frame_window_dif.reshape(1, *frame_window_dif.shape)
            # print(frame_window_new.dtype)
            ### Predict action from model
            output = model.predict(frame_window_new)[0]           
            predict_ind = np.argmax(output)
            
            ### Check noise of action
            if output[predict_ind] < 0.55:
                new_action = -1 # no action(noise)
            else:
                new_action = predict_ind # action detect

            ### Use State Machine to delete noise between action(just for stability)
            ### RUN_STATE: normal state, change to wait state when action is changed
            if state == RUN_STATE:
                if new_action != previous_action: # action change
                    state = WAIT_STATE
                    start_time = time.time()     
                else:
                    if previous_action == -1:
                        text_show = 'no action'                                              
                    else:
                        text_show = "{: <22}  {:.2f} ".format(class_text[previous_action],
                                    output[previous_action] )
                    print(text_show)  

            ### WAIT_STATE: wait 0.5 second when action from prediction is change to fillout noise
            elif state == WAIT_STATE:
                dif_time = time.time() - start_time
                if dif_time > 0.5: # wait 0.5 second
                    state = RUN_STATE
                    previous_action = new_action

            ### put text to image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text_show, (10,450), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)   
            
            ### shift sliding window
            frame_window = frame_window[1:n_sequence]
            
            ### To show dif RGB image
            # vis = np.concatenate((new_f, frame_window_new[0,n_sequence-1]), axis=0)
            # cv2.imshow('Frame', vis)
            cv2.imshow('Frame', frame)
        
    if cv2.waitKey(25) & 0xFF == ord('q'):
            break 
 
cap.release()
cv2.destroyAllWindows()
    