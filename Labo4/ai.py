from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import numpy as np

# data reading

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

# Model network

network = Sequential()

network = models.Sequential()
network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(128, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(128, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Flatten())
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))


network.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# Training

history = network.fit(train_generator, validation_data=validation_generator,epochs=5, batch_size=128)

# saving

network.save('model_cat_dog.h5');

# validating
tics = range(1, 1 +len(history.history['accuracy']))

plt.plot(tics, history.history['val_accuracy'], 'b', label='Validation acc')
plt.plot(tics, history.history['accuracy'], 'bo', label='Trainingsnauwkeurigheid')
plt.title('Nauwkeurigheid van training en validatie')
plt.legend()
plt.show()

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "C:\\Users\\boets\\OneDrive\\Documents\\GitHub\\NeuralNetworks\\Labo4\\test1", target_size=(50, 50), batch_size=20, class_mode='binary')

print(network.evaluate(test_generator))