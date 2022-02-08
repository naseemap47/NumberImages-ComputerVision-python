import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers


def display_some_samples(examples, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        idx = np.random.randint(0, examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]
        plt.subplot(5, 5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()

model = tf.keras.Sequential([
    layers.Input(shape=[28, 28, 1]),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D(),
    layers.BatchNormalization(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D(),
    layers.BatchNormalization(),

    layers.GlobalAvgPool2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("x_train Shape = ", x_train.shape)
    print("y_train Shape = ", y_train.shape)
    print("x_test Shape = ", x_test.shape)
    print("y_test Shape = ", y_test.shape)

    display_some_samples(x_train, y_train)