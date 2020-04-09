## Lets test on some images

from keras.models import load_model
import cv2
import re
import os
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import  ImageDataGenerator

model = load_model("/emo_det.h5")


num_classes = 6
img_width, img_height = 48, 48
batch_size = 16


validation_data_dir = "/fer2013/validation"

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                   color_mode='grayscale',
                                                   target_size=(img_width,img_height),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=False)


class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in  class_labels.items()}
classes = list(class_labels.values())


def draw_test(name, pred, im, true_label):
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 160, 0, 0, 300, cv2.BORDER_CONSTANT, value=BLACK)
    cv2.putText(expanded_image, "predicted -"+ pred, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    cv2.putText(expanded_image, "true -"+ true_label, (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    cv2.imshow(name, expanded_image)


def getRandomImage(path, img_width, img_height):
    # function loads a random images from a random folder in our path
    
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0, len(folders))
    path_class = folders[random_directory]
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0, len(file_names))
    image_name = file_names[random_file_index]
    final_path = file_path+"/"+image_name
    
    return image.load_img(final_path, target_size=(img_width, img_height), grayscale=True),final_path,path_class



files = []
predictions = []
true_labels =[]


#predicting images
for i in range(0,10):
    path = "/fer2013/validation/"
    img, final_path, true_label = getRandomImage(path, img_width, img_height)
    files.append(final_path)
    true_labels.append(true_label)
    
    x = image.img_to_array(img)
    x = x * 1./255
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict_classes(images,batch_size=10)
    predictions.append(classes)
    










