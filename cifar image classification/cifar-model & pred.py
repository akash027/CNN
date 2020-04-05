from __future__ import  print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D,MaxPooling2D

batch_size = 32
num_classes = 10
epochs = 1

#load the CIFAR dataset

(x_train, y_train), (x_test, y_test)  = cifar10.load_data()


print("shape of x_train: ", x_train.shape)
print(x_train.shape[0], " train_samples")
print(x_test.shape[0], " test_samples")

#format training data by Normalize and changing data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


## One Hot Encoding Our Lables(Y)

from keras.utils import np_utils

#now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


#Model

model = Sequential()

#padding = 'same' results in padding the input such that
#the output has the same length as the original input

model.add(Conv2D(32,(3,3), padding='same', 
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(64,(3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#initate RMSprop  optimizer and configuration some parameters
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    
print(model.summary())

# Train Model

history = model.fit(x_train,y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data = (x_test,y_test),
                    shuffle=True,verbose=1)

model.save('/home/sky/Documents/cifar10_model.h5')

scores = model.evaluate(x_test, y_test, verbose=1)
print("'Test loss: ", scores[0])
print("'Test Accuracy: ", scores[1])
