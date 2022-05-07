from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
from keras.models import Sequential
from keras import models
from keras import layers
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import rmsprop_v2

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.3)
])
model = models.Sequential()
model.add(data_augmentation)
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc'])

train_generator = train_datagen.flow_from_directory(
    "C:/Users/Cedric/Documents/GitHub/Python/Neural_Networks/Week4/cats_vs_dogs/train",
    class_mode='binary',
    batch_size=20,
    target_size=(150,150),
)

validation_generator = test_datagen.flow_from_directory(
    "C:/Users/Cedric/Documents/GitHub/Python/Neural_Networks/Week4/cats_vs_dogs/validation",
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('C:/Users/Cedric/Documents/GitHub/Python/Neural_Networks/Week4')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()