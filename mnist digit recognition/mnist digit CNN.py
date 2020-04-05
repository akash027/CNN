##Loading our dataset

from keras.datasets import mnist

#loads the mnist dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()


## Examine the size and image dimensions(for just practice)

print("Initial shape or dimensions of x_train: ", str(x_train.shape))

print("Number of samples in our training data: ", str(len(x_train)))
print("Number of lables in our training data: ", str(len(y_train)))

print("Number of lables in our testing data: ", str(len(y_test)))
print("Number of lables in our testing data: ", str(len(y_test)))
print()
print("Dimension of images: ", str(x_train[0].shape))
print("labels: ", str(y_train.shape))


#take look at some of images
#Using OpenCV

import numpy as np
import cv2

#display 6 random images from dataset

for i in range(0,6):
    random_num = np.random.randint(0, len(x_train))
    img = x_train[random_num]
    window_name = "Rnadom Samples #"+str(i)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)

cv2.destroyAllWindows()


# lets do it tha same using matplot (just for fun practices)

import matplotlib.pyplot as plt

#plots 6 images, NOTE subplot's arguments are nrows, ncols, index
#we set the color map to grey since our image dataset is grayscale
plt.subplot(331)
random_num = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

plt.subplot(332)
random_num = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

plt.subplot(333)
random_num = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

plt.subplot(334)
random_num = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

plt.subplot(335)
random_num = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

plt.subplot(336)
random_num = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

plt.show()



## Prepare our dataset for training

#lets store the num of rows and cols

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

#getting our data in the right shape needed for keras
#we need to add a 4th dimension to our data thereby changing our
#original image shape of (60000, 28, 28) to (60000, 28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)


#store the shape of a single image
input_shape = (img_rows, img_cols, 1)

#change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# Normalize the data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255

print("x_train shape: ", x_train.shape)


## One Hot Encoding Our Lables(Y)

from keras.utils import np_utils

#now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#count number of columns in our hot encode matrix
print("Num of classes:", str(y_train.shape[1]))

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]



## Creating our Model

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(0.01),
              metrics=['accuracy'])

print(model.summary())


## Train model

batch_size = 32
epochs =2

history = model.fit(x_train, y_train, 
                    batch_size= batch_size, 
                    epochs=epochs, verbose = 1,
                    validation_data = (x_test, y_test))



## Ploting Loss and Accuracy charts

import matplotlib.pyplot as plt

history_dict = history.history

loss_value = history_dict['loss']
val_loss_value = history_dict['val_loss']
epochs= range(1, len(loss_value)+1)

line1 = plt.plot(epochs, val_loss_value, label="Validation/test loss") 
line2 = plt.plot(epochs, loss_value, label="Trainning loss") 
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()


#Accuracy charts

acc_value = history_dict['accuracy']
val_acc_value = history_dict['val_accuracy']
epochs= range(1, len(acc_value)+1)

line1 = plt.plot(epochs, val_acc_value, label="Validation/test acc") 
line2 = plt.plot(epochs, acc_value, label="Trainning acc") 
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()



## Saving Model

model.save("/mnist_digits.h5")

## Loading model

from keras.models import  load_model

classifier = load_model("/mnist_digits.h5")



## Lets input some of our data into classifier

def draw_test(name, pred, input_im):
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(input_im, 0,0,0, imageL.shape[0],cv2.BORDER_CONSTANT, value = BLACK)
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_BGR2GRAY)
    cv2.putText(expanded_image, str(pred), (152,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0,255,0), 2)
    cv2.imshow(name,expanded_image)

for i in range(0,10):
    rand = np.random.randint(0,len(x_test))
    input_im = x_test[rand]
    
    imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
    input_im = input_im.reshape(1,28,28,1)
    
    #Get prediction
    res = str(classifier.predict_classes(input_im, 1, verbose=0)[0])
    
    draw_test("prediction",res, imageL)
    cv2.waitKey(0)

cv2.destroyAllWindows()


