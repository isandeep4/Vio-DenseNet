# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:15:21 2021

@author: isand
"""
import os
import numpy as np

def get_label(label):
    if(label == 'fight'):
        label = 1
    else:
        label = 0
    return label

train_test_lbls = os.listdir('./data/')
for train_test_lbl in train_test_lbls:
    with open('./data/{}.txt'.format(train_test_lbl),'w') as file:
        path = os.path.join('./data/',train_test_lbl)
        if os.path.isdir(path):
            lbls = os.listdir(path)
            for label in lbls:
                file_names = os.listdir(os.path.join(path,label))
                for file_name in file_names:
                    file.write(label)
                    file.write('/'+file_name)
                    file.write(' ')
                    lb = get_label(label)
                    file.write(str(lb) + '\n')
                    
with open('./data/test_new.txt','w') as file:
    path = os.path.join('./data', 'Test_data')
    print(path)
    labels = os.listdir(path)
    for label in labels:
        filespath = os.path.join(path,label)
        print(filespath)
        file_names = os.listdir(filespath)
        for filename in file_names:
            file.write(label)
            file.write('/'+ filename)
            file.write(' ')
            lb = get_label(label)
            file.write(str(lb)+ '\n')
            
            
        

