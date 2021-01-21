# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 11:25:00 2021

@author: isand
"""
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from data_generator import DataGenerator
from model import Create_pretrained_model
from data_helper import readfile_to_dict

dim = (224,224)
n_sequence = 8
n_channels = 3
n_output = 2
batch_size = 2
n_mul_test = 2
path_dataset = './data/train_data/'
weights_path = './save_weight/weight-39-0.97-0.94.hdf5' 
######

params = {'dim': dim,
          'batch_size': batch_size,
          'n_sequence': n_sequence,
          'n_channels': n_channels,
          'path_dataset': path_dataset,
          'option': 'RGBdiff',
          'shuffle': False}

test_txt = "./data/test_new.txt"
test_d = readfile_to_dict(test_txt)
key_list = list(test_d.keys()) * n_mul_test  # IDs


predict_generator = DataGenerator(key_list,test_d,**params,type_gen='predict')
model=Create_pretrained_model(dim,n_sequence,n_channels,n_output)
model.load_weights(weights_path)

y_pred_generator = model.predict_generator(predict_generator,workers=0)
test_y = np.array(list(test_d.values()) * n_mul_test)
print(y_pred_generator)
print(test_y)

y_pred = np.argmax(y_pred_generator,axis=1)
normalize = True
print(y_pred)

all_y = len(test_y)
sum = all_y
for i in range(len(y_pred)):
    if test_y[i] != y_pred[i]:
        sum -= 1
        print(key_list[i], 'Actual:', test_y[i], 'Predicted', y_pred[i])

cm = confusion_matrix(test_y, y_pred)
if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
else:
    print('Confusion matrix, without normalization')

accuracy = sum / all_y
print("accuracy:",accuracy)

classes = [*range(1,3)]

df_cm = pd.DataFrame(cm, columns=classes, index=classes)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
fig, ax = plt.subplots(figsize=(5,5))
sn.set(font_scale=0.6)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,fmt=".2f", annot_kws={"size": 8})# font size
# ax.set_ylim(5, 0)
plt.show()











