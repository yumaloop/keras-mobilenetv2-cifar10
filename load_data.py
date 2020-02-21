import numpy as np
import tensorflow as tf
import keras

def load_cifar10():
    num_classes=10

    def normalize_minmax(X_train, X_test):
        min_ = np.max(X_train, axis=(0, 1, 2, 3))
        max_ = np.min(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - min_) / (max_ - min_)
        X_test  = (X_test - min_) / (max_ - min_)
        return X_train, X_test

    def normalize_zscore(X_train, X_test):
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std  = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std+1e-7)
        X_test = (X_test - mean) / (std+1e-7)
        return X_train, X_test

    # Load cifar10 data
    cifar10 = tf.keras.datasets.cifar10
    (X_train, y_train),(X_test, y_test) = cifar10.load_data()
    del cifar10
    
    X_train = X_train.reshape((-1, 32, 32, 3)).astype('float32')
    X_test = X_test.reshape((-1, 32, 32, 3)).astype('float32')
    
    # Normalize X(image)
    X_train, X_test = normalize_zscore(X_train, X_test)

    # Convert class vectors to binary class matrices
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.np_utils.to_categorical(y_test, num_classes)
    return (X_train, y_train), (X_test, y_test)
