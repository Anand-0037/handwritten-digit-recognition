import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing now
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

model.save('handwritten-digit.keras')

# now we can load the model and test it without using above code again 
# as we have saved the model
# just comment the above code and run this part to load the model

# model = tf.keras.models.load_model("handwritten-digit.model")

# we can evaluate model.
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# now we can make our own directory and put images in it
# we can do it manually using writing digits on paint tool.
# do save them in format digit1.png, digit2.png in digits directory in handwritten-digit-recognition.
# then we can run the following code to predict the digits in those images.


# image_number =1
# while os.path.isfile(f"digits/digit{image_number}.png"):
#     try:
#         img =cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
#         img = np.invert(np.array([img]))
#         prediction = model.predict(img)
#         print(f"This digit is probably a {np.argmax(prediction)}")
#         plt.imshow(img[0], cmap = plt.cm.binary)
#         plt.show()
#     except:
#         print("Error")
#     finally:
#         image_number += 1


