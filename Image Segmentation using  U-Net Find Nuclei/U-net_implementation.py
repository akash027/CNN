import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


seed = 2020
random.seed = seed
np.random.seed =seed
tf.seed = seed



# Data Generator

class DataGen(keras.utils.Sequence):
    
    def __init__(self, ids, path, batch_size=8, image_size=128):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        
    
    def __load__(self, id_name):
        image_path = os.path.join(self.path, id_name, "images",  id_name) + ".png"
        mask_path = os.path.join(self.path, id_name, "masks/")
        all_masks = os.listdir(mask_path)
        
        image = cv2.imread(image_path,1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        mask = np.zeros((self.image_size, self.image_size, 1))
        
        for name in all_masks:
            _mask_path = mask_path + name
            _mask_image = cv2.imread(_mask_path, -1)
            _mask_image = cv2.resize(_mask_image, (self.image_size,self.image_size))
            _mask_image = np.expand_dims(_mask_image, axis=-1)
            mask = np.maximum(mask, _mask_image)
        
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
            
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            
            image.append(_img)
            mask.append(_mask)
        
        image = np.array(image)
        mask = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))




## Hyperparameters

image_size = 128
train_path = "/U_NET/train/" 
epochs = 5
batch_size=8

train_ids = next(os.walk(train_path))[1]

val_data_size = 10

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]



gen = DataGen(train_ids, train_path, batch_size=batch_size,image_size=image_size)
X, y = gen.__getitem__(index=0)


print(X.shape)
print(y.shape)





'''
U-NET  architecture has 3 parts
1. The Contracting/Downsampling Path
2. Bottleneck
3, The Expanding/Upsampling Path

Downsampling path:
    1.It consist of two 3x3 convolution(unpadded)
    2.At each downsampling step we double the number of feature channel...

Upsampling:
    Every step in the expensive path consist of an upsampling of the
    feature map followed by 2x2 conv. (up-sampling), a concatenation
    with the correspondingly feature map from the downsampling path....
   
Skip Connection:
    The skip connection from downsampling path are concatenated with feature map
    during upsampling path
    
Final Layer:
    At the final layer a 1x1 conv. is used to map each feature vector to the desired
    number of classes
'''



###----------- Building MODEL -----------###


from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import  Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


inputs_ = Input(shape = (image_size,image_size,3))

c1 = Conv2D(16,(3,3),activation='elu', kernel_initializer='he_normal',padding='same')(inputs_)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16,(3,3),activation='elu', kernel_initializer='he_normal',padding='same')(c1)
p1 = MaxPooling2D((2,2))(c1)

c2 = Conv2D(32,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(c2)
p2 = MaxPooling2D((2,2))(c2)

c3 = Conv2D(64,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(c3)
p3 = MaxPooling2D((2,2))(c3)

c4 = Conv2D(128,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(c4)
p4 = MaxPooling2D((2,2))(c4)


c5 = Conv2D(256,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(c5)

u6 = Conv2DTranspose(128,(2,2), strides=(2,2), padding='same')(c5)
u6 = concatenate([u6,c4])
c6 = Conv2D(128,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(c6)

u7 = Conv2DTranspose(64,(2,2), strides=(2,2), padding='same')(c6)
u7 = concatenate([u7,c3])
c7 = Conv2D(64,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(c7)

u8 = Conv2DTranspose(32,(2,2), strides=(2,2), padding='same')(c7)
u8 = concatenate([u8,c2])
c8 = Conv2D(32,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(c8)

u9 = Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(c8)
u9 = concatenate([u9,c1],axis=3)
c9 = Conv2D(16,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16,(3,3), activation='elu', kernel_initializer = 'he_normal', padding='same')(c9)


#Note our output is effectively a mask oof 128 x 128

outputs_ = Conv2D(1,(1,1), activation='sigmoid')(c9)


model = Model(inputs=[inputs_], outputs=[outputs_])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
model.summary()






# training model

train_gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
valid_gen = DataGen(valid_ids, train_path, batch_size=batch_size, image_size=image_size)


train_steps = len(train_ids) // batch_size
valid_steps = len(valid_ids) // batch_size


model.fit_generator(train_gen, validation_data=valid_gen,
                    steps_per_epoch=train_steps,
                    validation_steps=valid_steps,
                    epochs=epochs)




model.save_weights("/u-net.h5")

x, y = valid_gen.__getitem__(1)

results = model.predict(x)

results = results > 0.5


#display the results

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

#true data
ax = fig.add_subplot(1,2,1)
ax.imshow(np.reshape(y[0]*255, (image_size,image_size)), cmap='gray')

#predicted data
ax = fig.add_subplot(1,2,2)
ax.imshow(np.reshape(results[0]*255, (image_size,image_size)), cmap='gray')




fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
#predicted data
ax = fig.add_subplot(1,2,1)
ax.imshow(np.reshape(y[1]*255, (image_size,image_size)), cmap='gray')

#predicted data
ax = fig.add_subplot(1,2,2)
ax.imshow(np.reshape(results[1]*255, (image_size,image_size)), cmap='gray')






