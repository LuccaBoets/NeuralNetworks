from keras.models import load_model
import numpy as np
import tensorflow as tf

model = load_model('model_cat_dog.h5')

img = tf.keras.utils.load_img(
    "C:\\Users\\boets\\OneDrive\\Documents\\GitHub\\NeuralNetworks\\Labo4\\test1\\1.jpg",
    grayscale=False,
    color_mode='rgb',
    target_size=(25,25),
    interpolation='nearest'
)

print(model.predict(img, np.array(img)))