import tensorflow as tf
# To generate GIFs
!pip install -q imageio
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

#Choose either or, the first is the handwritten digit dataset and the next is the Fashion MNIST
(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.mnist.load_data()
(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.fashion_mnist.load_data()


#Display a selected image 
image = training_images[9, :]
image = image.reshape(28,28,1)
plt.imshow(image[:, :, 0], cmap='gray')

print(training_images.shape)


train_images = training_images / 255
train_images = train_images.reshape(train_images.shape[0], 784)
train_labels = training_labels
validation_images = testing_images
validation_images = validation_images.reshape(validation_images.shape[0], 784) / 255
validation_labels = testing_labels

print(train_images.shape)
print(validation_images.shape)
print(train_labels.shape)
print(validation_labels.shape)



model = tf.keras.Sequential([
      #Upsampling portion of model
      tf.keras.layers.Dense(7*7*256, activation = 'relu', input_shape = (1,)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Reshape((7, 7, 256)),
      tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), activation = 'relu', padding='same', use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same',  use_bias=False, activation = 'tanh'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(784, activation = 'sigmoid')
])

model.summary()




# Compile Model. 
model.compile(loss = 'categorical_crossentropy' , optimizer='adam', metrics=['accuracy'])

# Train the Model
history = model.fit(x = train_labels, y = train_images, batch_size = 500, epochs=10, validation_data = (validation_labels, validation_images), verbose = 1, validation_steps = 1)

model.evaluate(validation_labels, validation_images)



#Plotting
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))


#Plot the loss and training accuracy. Training accuracy will be low (2-3%), this is actually good for this model. 
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()



#Here you can pass any number into model.predict and it will output an image associated with that label.
prediction_og = model.predict([9])
prediction = prediction_og.reshape(28,28,1)
plt.imshow(prediction[:, :, 0], cmap='gray')
