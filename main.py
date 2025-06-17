import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing now
# uncomment it if you are running code for first time
# do comment out this code if handwritten-digit.keras file  is saved.

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

# model = tf.keras.models.load_model("handwritten-digit.keras")

# we can evaluate model.
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# now we can make our own directory and put images in it
# we can do it manually using writing digits on paint tool and then saving them.
# do save them in format digit1.png, digit2.png in digits directory in handwritten-digit-recognition.
# then we can run the following code to predict the digits in those images.


image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        # Load and preprocess image
        img = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))  # Ensure 28x28 size
        img = np.invert(img)  # Invert colors (white background to black)
        img = img / 255.0  # Normalize
        img = img.reshape(1, 28, 28)  # Reshape for model
        
        # Predict
        prediction = model.predict(img, verbose=0)  # verbose=0 suppresses progress bar
        confidence = np.max(prediction) * 100
        
        print(f"Image {image_number}: Predicted digit = {np.argmax(prediction)} (Confidence: {confidence:.1f}%)")
        
        # Display image
        plt.figure(figsize=(3, 3))
        plt.imshow(img[0], cmap='gray')
        plt.title(f"Predicted: {np.argmax(prediction)}")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error processing digit{image_number}.png: {e}")
    finally:
        image_number += 1

print(f"Processed {image_number-1} images total")

# do give name serial wise digit1.png and then 2,3,4.... in digits directory

