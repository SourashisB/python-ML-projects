import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data (scale pixel values to range [0,1])
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Input layer (Flatten 28x28 to 1D)
    keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons & ReLU activation
    keras.layers.Dropout(0.2),                   # Dropout layer to prevent overfitting
    keras.layers.Dense(10, activation='softmax') # Output layer (10 classes) with softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(x_test)

# Function to display an image and prediction
def plot_image(index):
    plt.imshow(x_test[index], cmap=plt.cm.binary)
    plt.title(f"Predicted: {np.argmax(predictions[index])}, Actual: {y_test[index]}")
    plt.show()

# Show a sample prediction
plot_image(0)  # Change index to see different results