import numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
import random
import  tensorflow as tf
import keras



# Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

plt.imshow(X_train[0], cmap='gray')

print(X_train.shape)
print(X_test.shape)


# Data visualization
i = random.randint(1,60000)
plt.imshow(X_train[i], cmap='gray')
plt.title(y_train[i])


# view more images in a grid format
# define dimensions of plot grid

W_grid = 15
L_grid = 15

#subplotd return the figure objects and axes objects
#we can use the axes objects to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize=(17,17))

axes = axes.ravel()  # flaten the 15 x 15 matrix into 255 array

n_training = len(X_train)


# select a random number from 0 to  n_training

for i in np.arange(0, W_grid*L_grid):
    index = np.random.randint(0, n_training)
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index], fontsize=8)
    axes[i].axis('off')
plt.show()




## Data preprocessing

#normalize data

X_train = X_train / 255
X_test = X_test / 255


# Add some noise 

noise_factor = 0.3

noise_dataset = []

for img in X_train:
    noisy_img = img + noise_factor * np.random.randn(*img.shape)
    noisy_img = np.clip(noisy_img, 0,1)
    noise_dataset.append(noisy_img)


plt.imshow(noise_dataset[10],cmap='gray')

noise_dataset = np.array(noise_dataset)


# Add some noise  to test data

noise_test_dataset = []

for img in X_test:
    noisy_img = img + noise_factor * np.random.randn(*img.shape)
    noisy_img = np.clip(noisy_img, 0,1)
    noise_test_dataset.append(noisy_img)


plt.imshow(noise_test_dataset[10],cmap='gray')

noise_test_dataset = np.array(noise_test_dataset)


# Building Autoencoder model


autoencoder = tf.keras.models.Sequential()

#encoder
autoencoder.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2,
                                       padding='same', input_shape=(28,28,1)))
autoencoder.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2,
                                       padding='same'))

autoencoder.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1,
                                       padding='same'))

#decoder
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2,
                                       padding='same'))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2,
                                       padding='same', activation='sigmoid'))


autoencoder.compile(loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(lr=0.001))
autoencoder.summary()



autoencoder.fit(noise_dataset.reshape(-1, 28,28,1),
                X_train.reshape(-1, 28,28,1),
                epochs=10,batch_size=200,
                validation_data=(noise_test_dataset.reshape(-1, 28,28,1),
                                X_test.reshape(-1, 28,28,1))
                )





# Evaluate training model performance

evaluation = autoencoder.evaluate(noise_test_dataset.reshape(-1, 28,28,1), X_test.reshape(-1, 28,28,1))

print('test loss: ', evaluation)



predicted = autoencoder.predict(noise_test_dataset.reshape(-1, 28,28,1))


fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True,sharey=True, figsize=(20,4))
for images, row in zip([noise_test_dataset[:10], predicted], axes):
    for img , ax in zip(images, row):
        ax.imshow(img.reshape((28,28)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
                  













