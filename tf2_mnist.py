import numpy as np
import tensorflow as tf
from keras.datasets import mnist

#Hyper parameter declaration
num_features = 784
output_size = 10
hidden_layer1_size = 128
hidden_layer2_size = 256
num_epochs = 5
batch_size = 100
learning_rate = 0.01

# Loading mnist hand written dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Convert to float32 to make sure and convert to float32

x_train, x_test = np.array(X_train, np.float32), np.array(X_test, np.float32)

# Flatten images to 3-D vector of 784 features as (28*28*1).
X_train, X_test = X_train.reshape([-1, 28, 28, 1]), X_test.reshape([-1, 28, 28, 1])

X_train, X_test = X_train / 255., x_test / 255.

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(hidden_layer1_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer2_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, epochs=num_epochs, steps_per_epoch=batch_size, verbose=2)
