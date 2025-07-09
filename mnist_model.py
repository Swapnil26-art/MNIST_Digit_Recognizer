# 1. Import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input

# 2. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Shape of training data:", x_train.shape)

# 3. Normalize the image pixel values to [0, 1] range
x_train = x_train / 255.0
x_test = x_test / 255.0

# 4. Reshape data to add channel dimension (for CNN input: [batch, height, width, channels])
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 5. One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 6. Build the CNN model (ONLY ONCE)
model = Sequential([
    Input(shape=(28, 28, 1)),           # Input layer
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')    # Output layer for 10 classes
])

# 7. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 8. Train the model
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# 9. Evaluate the model on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)

# 10. Save the model (for later UI use)
model.save("mnist_cnn_model.h5")
