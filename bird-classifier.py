#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical

# Load local images from directory
train_dir = "/Users/zubairfaruqui/Downloads/CNN Data/Training Dataset"  # Specify the path to the training folder
test_dir = "/Users/zubairfaruqui/Downloads/CNN Data/Test Dataset"    # Specify the path to the testing folder

batch_size = 32
img_size = (960, 480)  # MNIST-like images

# Load train and test datasets from the directories, in grayscale mode
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,  # Images resized to match the MNIST dimensions
    batch_size=batch_size,
    color_mode='grayscale',  # Load as grayscale images
    label_mode='int'  # Labels will be integers corresponding to folder names
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',  # Load as grayscale images
    label_mode='int'
)

# Convert datasets to NumPy arrays for use with the rest of the code
def dataset_to_numpy(dataset):
    images = []
    labels = []
    for image_batch, label_batch in dataset:
        images.append(image_batch.numpy())
        labels.append(label_batch.numpy())
    return np.concatenate(images), np.concatenate(labels)

X_train, y_train = dataset_to_numpy(train_dataset)
X_test, y_test = dataset_to_numpy(test_dataset)

# Normalize images (convert from range [0, 255] to [0, 1])
X_train = X_train / 255.0
X_test = X_test / 255.0

# Check shapes
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# Show some sample images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    num = random.randint(0, len(X_train))
    plt.imshow(X_train[num].squeeze(), cmap='gray', interpolation='none')  # Use .squeeze() to remove single channel dimension
    plt.title("Class {}".format(y_train[num]))

plt.tight_layout()

# Flatten the images from (28, 28, 1) to (784) for the neural network
X_train = X_train.reshape(X_train.shape[0], 960 * 480)
X_test = X_test.reshape(X_test.shape[0], 960 * 480)

# Convert labels to categorical (one-hot encoding)
no_classes = len(np.unique(y_train))
Y_train = to_categorical(y_train, no_classes)
Y_test = to_categorical(y_test, no_classes)

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(960 * 480,)))  # Input shape adjusted for flattened 28x28 grayscale images
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(no_classes))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, batch_size=128, epochs=1, verbose=1)

# Evaluate the model
score = model.evaluate(X_test, Y_test)
print('Test accuracy:', score[1])

# Plot accuracy and loss
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

# Predictions on test data
predicted_classes = np.argmax(model.predict(X_test), axis=1)

correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

# Visualize correct predictions
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(960, 480, i + 1)
    plt.imshow(X_test[correct].reshape(960, 480), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))

plt.tight_layout()

# Visualize incorrect predictions
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(960, 480, i + 1)
    plt.imshow(X_test[incorrect].reshape(960, 480), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))

plt.tight_layout()
