#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:41:11 2018

@author: dmitri
"""

#PAC2018_0007.nii
import os
import numpy as np
import pickle
import nibabel as nb
import pandas as pd
from scipy.stats import zscore
from keras import layers, models, optimizers, regularizers
from random import shuffle

outlier_thr = 5
# Read csv file
df = pd.read_csv('PAC2018_Covariates_detailed.csv')

# extract relevant variables
sub_id = np.array(df['PAC_ID'])
label = np.array(df['Label'] - 1)
df = df.drop(['PAC_ID', 'Label'], 1)
header = df.keys()

# Clean dataset - drop subjects with values above `outlier_thr` STD
outliers = np.sum((np.abs(zscore(df)) > outlier_thr), 1) != 0
print('%d outliers detected.' % outliers.sum())
data = np.array(df.drop(np.where(outliers)[0]))
sub_id = sub_id[np.invert(outliers)]
label = label[np.invert(outliers)]

# zscore data
data = zscore(data)

# Reset Gender and Scanner values to nominal values
data[:,1] = (data[:,1]>0) + 1
data[:,3] = [np.where(i==np.unique(data[:,3]))[0][0] + 1for i in data[:,3]]

data, sub_id, label = pd.DataFrame(data, columns=header), sub_id, label


def balance_dataset(sub_id, labels, data):
    max_label_size = np.min([np.sum(lab == labels) 
                             for lab in np.unique(labels)])

    labels_1 = np.where(labels == 0)[0]
    np.random.shuffle(labels_1)
    labels_1 = labels_1[:max_label_size]

    labels_2 = np.where(labels == 1)[0]
    np.random.shuffle(labels_2)
    labels_2 = labels_2[:max_label_size]

    new_data_id = np.hstack((labels_1, labels_2))
    np.random.shuffle(new_data_id)
    labels = labels[new_data_id]
    sub_id = sub_id[new_data_id]
    data = data[new_data_id]

    return (sub_id, labels, data)


def get_train_valid_set(sub_id, label, data, group='123', train_ratio=0.8):
    
    selecter = [str(int(d)) in group for d in data.Scanner]

    group_sub, group_label, group_data = balance_dataset(
        sub_id[selecter], label[selecter], np.array(data[selecter]))
    
    
    train_size = int(len(group_sub) * train_ratio)
    valid_size = len(group_sub) - train_size

    counter1 = 0
    counter2 = 0
    train_list = []

    for i, s in enumerate(group_sub):
        if counter1 < (train_size / 2) and group_label[i] == 0:
            train_list.append(s)
            counter1 += 1
        elif counter2 < (train_size / 2) and group_label[i] == 1:
            train_list.append(s)
            counter2 += 1

    selecter = np.array([True if e in train_list else False for i, e in enumerate(group_sub)])

    train_list = group_sub[selecter]
    valid_list = group_sub[np.invert(selecter)]
    
    return train_list, valid_list, group_sub, group_label, group_data


def data_gen(fileList, batch):

    while True:
        for r in range(0, len(fileList), batch):

            batch_data = []
            batch_label = []

            for i in range(batch):
                if r + i >= len(fileList):
                    break
                else:

                    patientID = fileList[r]
                    f = 'nifti/%s.nii' % patientID

                    # Get data for each subject
                    img = nb.load(f).get_fdata()
                    img = img[15:105, 15:125, 15:100]

                    batch_data.append(img)

                    # Get data for each label
                    labelID = group_label[group_sub == patientID]
                    batch_label.append(labelID)

            yield (np.array(batch_data)[..., None], np.array(batch_label))
            
            
            
# Neural network definition
model = models.Sequential()

input_shape = (90, 110, 85, 1)

# Convolutions and Pooling
"""
model.add(layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', strides=(1, 1, 1),
                        input_shape=input_shape, batch_size=None))
model.add(layers.MaxPooling3D(pool_size = (3,3,3)))
model.add(layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', strides=(1, 1, 1)))
model.add(layers.MaxPooling3D(pool_size=(3, 3, 3)))

model.add(layers.Conv3D(64, kernel_size=(2, 2, 2), activation='relu', strides=(1, 1, 1)))
model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

model.add(layers.Conv3D(64, kernel_size=(2, 2, 2), activation='relu', strides=(1, 1, 1)))
model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

# Flattening
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

98% overfitted
"""



model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', strides=(1, 1, 1),
                        input_shape=input_shape, batch_size=None))
model.add(layers.MaxPooling3D(pool_size = (3,3,3)))
model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', strides=(1, 1, 1)))
model.add(layers.MaxPooling3D(pool_size = (3,3,3)))
model.add(layers.Dropout(0.1))
model.add(layers.Conv3D(16, kernel_size=(2, 2, 2), activation='relu', strides=(1, 1, 1)))
model.add(layers.MaxPooling3D(pool_size = (3,3,3)))
model.add(layers.Conv3D(16, kernel_size=(2, 2, 2), activation='relu', strides=(1, 1, 1)))
model.add(layers.BatchNormalization())

# Flattening
model.add(layers.Flatten())
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
# Batch size
batch_size = 1
# Training to Validation set ratio
train_ratio=0.9
# Create Groups
group_id = '23'
train_list, valid_list, group_sub, group_label, group_data = get_train_valid_set(
    sub_id, label, data, group=group_id, train_ratio=0.8)
model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint
 
checkpoint = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)

history = model.fit_generator(
    data_gen(train_list, batch_size),
    callbacks = [checkpoint],
    steps_per_epoch=int(np.ceil(len(train_list) / batch_size)),
    validation_data=data_gen(valid_list, batch_size),
    validation_steps=int(np.ceil(len(valid_list) / batch_size)),
    epochs=200, shuffle=True)
                                 

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.plot(history.history['acc'], label='Training accuracy')
plt.plot(history.history['val_acc'], label='Testing accuracy')
plt.ylim([0,1])
plt.title('Training Accuracy vs Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc=4)
plt.tight_layout()
plt.show()
plt.savefig('accuracy.png')

plt.clf()
plt.style.use('fivethirtyeight')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Test Loss')
plt.legend(loc=4)
plt.tight_layout()
plt.show()
plt.savefig('loss.png')
