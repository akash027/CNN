import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import  Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import  cv2

batch_size = 16
epochs = 5

img = cv2.imread("/home/sky/Documents/2.CNN/Data augmentation-catVSdog/catsvsdogs/train/cats/cat001.jpg")

print(img.shape)

img_rows = 150
img_cols = 150

input_shape = img_rows, img_cols, 3


train_dir = "/home/sky/Documents/2.CNN/Data augmentation-catVSdog/catsvsdogs/train/"
validation_dir = "/home/sky/Documents/2.CNN/Data augmentation-catVSdog/catsvsdogs/validation/"

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = train_datagen.flow_from_directory(validation_dir,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='binary')


num_train_samples = 2000
num_valid_samples = 1000


### MODEL

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


print(model.summary())


model.fit_generator(train_generator,
                    steps_per_epoch = num_train_samples // batch_size,
                    epochs = epochs,
                    validation_data = validation_generator,
                    validation_steps = num_valid_samples // batch_size)
