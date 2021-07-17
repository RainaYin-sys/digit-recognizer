#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 22:53:43 2021

@author: yinxiaoru
"""

# 加载数据库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.optimizers import Adam

# 数据准备 
X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
Y_train = X_train['label']
X_train = X_train.drop(['label'],axis=1)
#X_train.head()
#X_test.head()
#print(X_train.shape,X_test.shape)

# 把数据整理成3维
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
#print(X_train.shape,X_test.shape)
#print(np.max(X_train),np.min(X_train))

# 查看一张图片
#fig1 = plt.figure(figsize=(8,12))
#plt.imshow(X_train[16,:,:,:])

# 对 label 进行 one-hot-coding 
Y_train = to_categorical(Y_train,10)

# 归一化处理
X_train = X_train / 255.0
X_test = X_test / 255.0

# 分离训练集和测试集
random_seed = 2
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.1,random_state=random_seed)

#搭建模型
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(200,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(10,activation='softmax'))

# 定义模型优化器
adam = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

# 训练模型
model.fit(X_train,Y_train,batch_size=100,epochs=1,verbose=1,validation_data=[X_val,Y_val])

# 模型存储
model.save('my_Alexnet_mnist.h5')

results = model.predict(X_test)
results = np.argmax(results,axis=1)
results = pd.Series(results,name = 'Label')
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis=1)
submission.to_csv('cnn_minist_datagen.csv')


















