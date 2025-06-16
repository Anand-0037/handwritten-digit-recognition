import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing now
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model= tf.keras.models.Sequential()
model.add(tf.keras.layes.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layesr.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save("handwritten-digit.model")

# now we can load the model and test it without using above code again 
# as we have saved the model
# just comment the above code and run this part to load the model

# new_model = tf.keras.models.load_model("handwritten-digit.model")

# we can evaluate model.
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# now we can make our own directory and put images in it
# we can do it manually using writing digits on paint tool.

