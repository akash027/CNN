import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os



num_classes = 6
img_rows, img_cols = 48, 48
batch_size = 16

train_data_dir = "/fer2013/train"
validation_data_dir = "/fer2013/validation"


#lets use some data augmentation

train_datgen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,
                                  shear_range=0.3,
                                  zoom_range=0.3,
                                  width_shift_range=0.4,
                                  height_shift_range=0.4,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datgen.flow_from_directory(train_data_dir,
                                                   color_mode='grayscale',
                                                   target_size=(img_rows,img_cols),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=True)


validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                   color_mode='grayscale',
                                                   target_size=(img_rows,img_cols),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=True)


# model like littlwVGG 

model = Sequential()

model.add(Conv2D(32,(3,3), padding='same', kernel_initializer='he_normal',
                 input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3), padding='same', kernel_initializer='he_normal',
                 input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# Block #2: second CONV -> ELU -> CONV -> RELU -> POOL
# layer set
model.add(Conv2D(64,(3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# Block #3: third CONV -> ELU -> CONV -> RELU -> POOL
# layer set
model.add(Conv2D(128,(3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# Block #4: fourth CONV -> ELU -> CONV -> RELU -> POOL
# layer set
model.add(Conv2D(256,(3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))


# Block #5:  first set of FC -> ELU layers
model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# Block 6: second set of Fc -> ELu layes
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# Block 7: softmax classifier
model.add(Dense(num_classes, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())




from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


checkpoint = ModelCheckpoint("/emo_det.h5",
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=2,
                              min_delta=0.0001)



#we put our callbacks into a callback list
callbacks = [earlystop, checkpoint, reduce_lr]


#we use a very small learning rate
model.compile(loss='categorical_crossentropy',
              optimizer= Adam(lr=0.001),
              metrics=['accuracy'])


nb_train_samples = 28273
nb_validation_samples = 3534

epochs = 10


history = model.fit_generator(train_generator, epochs=epochs,
                              steps_per_epoch=nb_train_samples // batch_size,
                              callbacks=callbacks,
                              validation_data = validation_generator,
                              validation_steps = nb_validation_samples // batch_size)






import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import  classification_report, confusion_matrix
import numpy as np


# we need to recreate our validation generator with shuffle=False


validation_generator_s_f = validation_datagen.flow_from_directory(validation_data_dir,
                                                   color_mode='grayscale',
                                                   target_size=(img_rows,img_cols),
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=False)


class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in  class_labels.items()}
classes = list(class_labels.values())


#confusion matrix and classification_report

Y_pred = model.predict_generator(validation_generator_s_f, nb_validation_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)


print("Confusion matrix")
print(confusion_matrix(validation_generator_s_f.classes,y_pred))


print("Classification_report")
target_names = list(class_labels.values())
print(classification_report(validation_generator_s_f.classes,y_pred,target_names=target_names))



plt.figure(figsize = (8,8))
cnf_matrix = confusion_matrix(validation_generator_s_f.classes,y_pred)

plt.imshow(cnf_matrix, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(classes))

_ = plt.xticks(tick_marks, classes, rotation=90)
_ = plt.yticks(tick_marks, classes)





