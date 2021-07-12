# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 17:59:05 2019

@author: richard
"""

import os
import copy
import sequence_global_variable as sgv
import posture_selection as ps
import numpy as np
from keras import optimizers
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

output_model = sgv.model_directory

if not os.path.exists(output_model):
    os.makedirs(output_model)

number_epoch = 300
# create plot for the model
def create_plot(hist, title, nb_epoch, path):
    xc = range(nb_epoch)
    a = hist.history[title]
    b = hist.history['val_'+title]
    plt.figure()
    plt.plot(xc, a)
    plt.plot(xc, b)
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.title('train_'+title+' vs val_'+title)
    plt.grid(True)
    plt.legend(['train', 'val'])
    plt.savefig(path)

training_data = ps.centroid_features
training_data = np.array(training_data)
class_data = ps.class_features
class_data = np.array(class_data)

cleaning_back_legs_data = copy.copy(training_data[0:145][:][:])
cleaning_back_legs_class = copy.copy(class_data[0:145][:])
train_cleaning_back_legs_X, valid_cleaning_back_legs_X, train_cleaning_back_legs_Y, valid_cleaning_back_legs_Y = train_test_split(cleaning_back_legs_data, cleaning_back_legs_class, test_size=0.33, shuffle=True)

cleaning_back_data = copy.copy(training_data[145:290][:][:])
cleaning_back_class = copy.copy(class_data[145:290][:])
train_cleaning_back_X, valid_cleaning_back_X, train_cleaning_back_Y, valid_cleaning_back_Y = train_test_split(cleaning_back_data, cleaning_back_class, test_size=0.33, shuffle=True)

cleaning_legs_data = copy.copy(training_data[290:435][:][:])
cleaning_legs_class = copy.copy(class_data[290:435][:])
train_cleaning_legs_X, valid_cleaning_legs_X, train_cleaning_legs_Y, valid_cleaning_legs_Y = train_test_split(cleaning_legs_data, cleaning_legs_class, test_size=0.33, shuffle=True)

cleaning_head_data = copy.copy(training_data[435:580][:][:])
cleaning_head_class = copy.copy(class_data[435:580][:])
train_cleaning_head_X, valid_cleaning_head_X, train_cleaning_head_Y, valid_cleaning_head_Y = train_test_split(cleaning_head_data, cleaning_head_class, test_size=0.33, shuffle=True)

cleaning_left_chest_data = copy.copy(training_data[580:725][:][:])
cleaning_left_chest_class = copy.copy(class_data[580:725][:])
train_cleaning_left_chest_X, valid_cleaning_left_chest_X, train_cleaning_left_chest_Y, valid_cleaning_left_chest_Y = train_test_split(cleaning_left_chest_data, cleaning_left_chest_class, test_size=0.33, shuffle=True)

cleaning_right_chest_data = copy.copy(training_data[725:870][:][:])
cleaning_right_chest_class = copy.copy(class_data[725:870][:])
train_cleaning_right_chest_X, valid_cleaning_right_chest_X, train_cleaning_right_chest_Y, valid_cleaning_right_chest_Y = train_test_split(cleaning_right_chest_data, cleaning_right_chest_class, test_size=0.33, shuffle=True)

cleaning_back_head_data = copy.copy(training_data[870:][:][:])
cleaning_back_head_class = copy.copy(class_data[870:][:])
train_cleaning_back_head_X, valid_cleaning_back_head_X, train_cleaning_back_head_Y, valid_cleaning_back_head_Y = train_test_split(cleaning_back_head_data, cleaning_back_head_class, test_size=0.33, shuffle=True)

temp_train_data_X = np.concatenate([train_cleaning_back_legs_X, train_cleaning_back_X, train_cleaning_legs_X, train_cleaning_head_X, train_cleaning_left_chest_X, train_cleaning_right_chest_X, train_cleaning_back_head_X])
temp_valid_data_X = np.concatenate([valid_cleaning_back_legs_X, valid_cleaning_back_X, valid_cleaning_legs_X, valid_cleaning_head_X, valid_cleaning_left_chest_X, valid_cleaning_right_chest_X, valid_cleaning_back_head_X])
temp_train_data_Y = np.concatenate([train_cleaning_back_legs_Y, train_cleaning_back_Y, train_cleaning_legs_Y, train_cleaning_head_Y, train_cleaning_left_chest_Y, train_cleaning_right_chest_Y, train_cleaning_back_head_Y])
temp_valid_data_Y = np.concatenate([valid_cleaning_back_legs_Y, valid_cleaning_back_Y, valid_cleaning_legs_Y, valid_cleaning_head_Y, valid_cleaning_left_chest_Y, valid_cleaning_right_chest_Y, valid_cleaning_back_head_Y])

index_random_train_data = np.arange(0, temp_train_data_X.shape[0])
np.random.shuffle(index_random_train_data)

index_random_valid_data = np.arange(0, temp_valid_data_X.shape[0])
np.random.shuffle(index_random_valid_data)

train_data_X = []
valid_data_X = []
train_data_Y = []
valid_data_Y = []

for i in range(temp_train_data_X.shape[0]):
    train_data_X.append(copy.copy(temp_train_data_X[index_random_train_data[i]]))
    train_data_Y.append(copy.copy(temp_train_data_Y[index_random_train_data[i]]))

for i in range(temp_valid_data_Y.shape[0]):
    valid_data_X.append(copy.copy(temp_valid_data_X[index_random_valid_data[i]]))
    valid_data_Y.append(copy.copy(temp_valid_data_Y[index_random_valid_data[i]]))
    
train_data_X = np.array(train_data_X)
valid_data_X = np.array(valid_data_X)
train_data_Y = np.array(train_data_Y)
valid_data_Y = np.array(valid_data_Y)

model = Sequential()

# Architecture 1 (300 Epoch)
#model.add(GRU(100, input_shape=(30, 25)))
#model.add(Dropout(0.3))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(7, activation='softmax'))

# Architecture 2 (300 Epoch)
#model.add(LSTM(100, input_shape=(30, 25)))
#model.add(Dropout(0.3))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(7, activation='softmax'))

# Architecture 3 (300 Epoch)
model.add(GRU(100, input_shape=(30, 25)))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax'))

opt = optimizers.SGD(lr=0.0001, momentum=0.9)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=1)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

tb = TensorBoard(histogram_freq=1, write_grads=True)

hist = model.fit(train_data_X, train_data_Y, validation_data=(valid_data_X, valid_data_Y), batch_size=8, epochs=number_epoch, verbose=2, callbacks=[tb, reduce_lr])
create_plot(hist, 'acc', number_epoch, "{}/model_acc_result_{}.png".format(output_model, number_epoch))
create_plot(hist, 'loss', number_epoch, "{}/model_loss_result_{}.png".format(output_model, number_epoch))

model_json = model.to_json()

with open('{}/model_{}.json'.format(output_model, number_epoch), "w") as json_file:
    json_file.write(model_json)
    json_file.close()

model.save_weights("{}/model_weight_{}.h5".format(output_model, number_epoch))


