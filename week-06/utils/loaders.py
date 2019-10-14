from keras.datasets import fashion_mnist


def load_mnist_fashion():
    # Load the dataset
    (X_train, _), (_, _) = fashion_mnist.load_data()
    return X_train

