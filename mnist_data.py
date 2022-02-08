import tensorflow as tf

if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("x_train Shape = ", x_train.shape)
    print("y_train Shape = ", y_train.shape)
    print("x_test Shape = ", x_test.shape)
    print("y_test Shape = ", y_test.shape)
