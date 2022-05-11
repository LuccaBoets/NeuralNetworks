from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import models, Sequential
from keras.models import load_model
from keras import layers
from keras import optimizers
from keras.optimizers import rmsprop_v2
import matplotlib.pyplot as plt

BASE_DIR = "cats_and_dogs_small"

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
    return model

def train_model(model: Sequential):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        BASE_DIR + "/train", target_size=(150, 150), batch_size=20, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
        BASE_DIR + "/validation", target_size=(150, 150), batch_size=20, class_mode='binary')

    history = model.fit(train_generator, steps_per_epoch=100,
                        epochs=30, validation_data=validation_generator, validation_steps=50)

    model.save('cats_and_dogs_small_1.h5')
    return history

def evaluate_model(model: Sequential):
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        BASE_DIR + "/test", target_size=(150, 150), batch_size=20, class_mode='binary')

    model.evaluate(test_generator)

def draw_graphs(history):
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
    return


def main():
    print("Creating model")
    model = create_model()
    print("Creating complete")

    print("Training model")
    history = train_model(model)
    print("Training complete")

    print("Evaluate model")
    evaluate_model(model)
    print("Evaluation complete")

    print("Drawing graphs")
    draw_graphs(history)
