from keras import models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import layers

def ModelAanmaken():
    network = models.Sequential()
    network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
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

    return network

def ModelTreinen(network, trainGen, validationGen, count):
    return network.fit(trainGen, validation_data=validationGen,epochs=count, batch_size=128)

def ToonGeschiedenis(geschiedenis):
    tics = range(1, 1 +len(geschiedenis.history['accuracy']))

    plt.plot(tics, geschiedenis.history['val_accuracy'], 'b', label = 'Validatie')
    plt.plot(tics, geschiedenis.history['accuracy'], 'bo', label = 'Treining')
    plt.title("Accuraatheid")
    plt.show()


trainData = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    )

testData = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    )

trainGen = trainData.flow_from_directory(
    "dataset\\train",
    target_size=(150, 150),
    batch_size=40,
    color_mode='grayscale',
    class_mode='binary')



validationGen = testData.flow_from_directory(
    "dataset\\test",
    target_size=(150, 150),
    batch_size=40,
    color_mode='grayscale',
    class_mode='binary')

network = ModelAanmaken()
network.save('model.h5');

geschiedenis = ModelTreinen(network, trainGen, validationGen, 2)
ToonGeschiedenis(geschiedenis)