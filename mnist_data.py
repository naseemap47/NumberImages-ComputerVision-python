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

    # To display some sample data
    if False:
        display_some_samples(x_train, y_train)
    
    # Normalize data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # In Model input we need 3 dimensional data
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    # Cmpliling the Model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics='accuracy'
    )
    
    # fit the Model - Model Training
    model.fit(
        x_train, y_train,
        validation_split=0.2,
        batch_size=64,
        epochs=3
    )