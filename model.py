# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:32:50 2021

@author: isand
"""

from keras import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.layers import LSTM, CuDNNLSTM
from keras.layers import Dropout, TimeDistributed
from keras.applications import MobileNetV2
from keras import backend as k
from tensorflow.python.keras import optimizers
from keras.models import Model

def Create_pretrained_model(dim, n_sequence, n_channels, n_output):
    model = Sequential()
    model.add(
        TimeDistributed(
            MobileNetV2(weights='imagenet',include_top=False), 
            input_shape=(n_sequence, *dim, n_channels)            
            )        
        )
    model.add(
        TimeDistributed(
            GlobalAveragePooling2D()
            )
        )
    model.add(CuDNNLSTM(64,return_sequences=False))
    model.add(Dense(64,activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(24,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_output,activation='softmax'))
    
    model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    return model