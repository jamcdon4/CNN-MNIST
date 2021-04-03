from __future__ import absolute_import, division, print_function, unicode_literals

import keras
import matplotlib.pyplot as plt
import os
import mnist
import numpy as np
import csv

from keras import backend as k
from keras import layers
from mnist import MNIST

img_rows, img_cols = 28, 28

mndata = MNIST('samples')

#load mnist dataset
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
(train_images, train_labels) = mndata.load_training()
(test_images, test_labels) = mndata.load_testing()

#index = random.randrange(0, len(test_images))  # choose an index ;-)
#print(mndata.display(test_images[index]))

#convert data read locally to match analysis
X_train = np.array(train_images)
X_train = X_train.reshape((-1,28,28))
y_train = np.array(train_labels)

X_test = np.array(test_images)
X_test = X_test.reshape((-1,28,28))
y_test = np.array(test_labels)


#plot data for verification / understanding
# fig = plt.figure()
# for i in range(9):
#   plt.subplot(3,3,i+1)
#   plt.tight_layout()
#   plt.imshow(X_train[i], cmap='gray', interpolation='none')
#   plt.title("Digit: {}".format(y_train[i]))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()


# reshaping
# this assumes our data format
# For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while
# "channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
#more reshaping
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape) #X_train shape: (60000, 28, 28, 1)


#set number of categories
num_category = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)


##model building
model = keras.Sequential()
#convolutional layer with rectified linear unit activation
model.add(layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#32 convolution filters used each of size 3x3
#again
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#64 convolution filters used each of size 3x3
#choose the best features via pooling
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#randomly turn neurons on and off to improve convergence
model.add(layers.Dropout(0.25))
#flatten since too many dimensions, we only want a classification output
model.add(layers.Flatten())
#fully connected to get all relevant data
model.add(layers.Dense(128, activation='relu'))
#one more dropout for convergence' sake :)
model.add(layers.Dropout(0.5))
#output a softmax to squash the matrix into output prob
model.add(layers.Dense(num_category, activation='softmax'))

#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
#categorical ce since we have multiple classes (10)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 128
num_epoch = 15
#model training
model_log = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0]) #Test loss: 0.0296396646054
print('Test accuracy:', score[1]) #Test accuracy: 0.9904

# output Results to mnist.csv
predictions = model.predict(X_test)
with open("mnist.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(predictions)

#plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_log.history['accuracy'])
plt.plot(model_log.history['val_accuracy'])
plt.title('Convolutional Tanh model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('Convolutional Tanh model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
plt.show()
fig