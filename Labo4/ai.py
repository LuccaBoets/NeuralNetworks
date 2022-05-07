from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
import keras
from keras.models import Sequential
from keras import models
from keras import layers
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import rmsprop_v2

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1
    )

test_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1
    )

train_generator = train_datagen.flow_from_directory(
    "C:\\Users\\boets\\OneDrive\\Documents\\GitHub\\NeuralNetworks\\Labo4\\dataset\\train",
    target_size=(50, 50),
    batch_size=40,
    class_mode='binary')



validation_generator = test_datagen.flow_from_directory(
    "C:\\Users\\boets\\OneDrive\\Documents\\GitHub\\NeuralNetworks\\Labo4\\dataset\\test",
    target_size=(50, 50),
    batch_size=40,
    class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('shape: ', data_batch.shape)
    break

network = Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=(50, 50, 3)))
network.add(layers.Flatten())
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

history = network.fit(train_generator, validation_data=validation_generator,epochs=5, batch_size=128)

network.save('model_cat_dog.h5');

network.validate()