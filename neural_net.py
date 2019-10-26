import numpy as np
from matplotlib import pyplot as plt

def sigmoid(Z):
    Z = 1 / (1 + np.exp(-Z))
    return Z


def relu(Z):
    Z = np.maximum(Z, 0)
    return Z


def relu_derivative(Z):
    deriv = np.zeros(Z.shape)
    deriv[Z > 0] = 1
    return deriv


def loss_function(Y_hat, train_data):
    loss = (-1 / train_data['m']) * np.sum(train_data['Y'] * np.log(Y_hat) + (1 - train_data['Y']) * np.log(1 - Y_hat))
    return loss


def forward_propagation(X: np.ndarray, weights: dict):
    Z1 = np.dot(weights['W1'], X) + weights['b1']
    A1 = relu(Z1)
    Z2 = np.dot(weights['W2'], A1) + weights['b2']
    A2 = sigmoid(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2}

    return A2, cache


def back_propagation(A2, weights, train_data, cache):
    dZ2 = A2 - train_data['Y']
    dW2 = (1 / train_data['m']) * np.dot(dZ2, cache['A1'].T)
    db2 = (1 / train_data['m']) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(weights['W2'].T, dZ2) * relu_derivative(cache['Z1'])
    dW1 = (1 / train_data['m']) * np.dot(dZ1, train_data['X'].T)
    db1 = (1 / train_data['m']) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {'dW1': dW1,
                 'db1': db1,
                 'dW2': dW2,
                 'db2': db2}

    return gradients


def initialize_weights(n_x: int, n_h: int):
    W1 = np.random.randn(n_h, n_x) * np.sqrt(2 / n_x)
    b1 = np.random.randn(n_h, 1)
    W2 = np.random.randn(1, n_h) * np.sqrt(2 / n_h)
    b2 = np.random.randn(1, 1)

    weights = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return weights


def update_weights(weights, gradients, learning_rate):
    weights['W2'] -= learning_rate * gradients['dW2']
    weights['b1'] -= learning_rate * gradients['db1']
    weights['W1'] -= learning_rate * gradients['dW1']
    weights['b2'] -= learning_rate * gradients['db2']
    return weights


def train_nn_binary_classifier(X: np.ndarray, Y: np.ndarray, n_h: int, learning_rate: float, iterations: int):
    """
    Trains neural network based on input data.

    :param X: 2d array of training data, samples should be stored in columns.
    :param Y: row vector of training data classes (0 or 1).
    :param n_h: number of neurons in hidden layer.
    :param learning_rate: learning rate used in gradient descent.
    :param iterations: number of iterations of gradient descent.
    :return: weights – a dictionary containing weights that can be used to predict using predict_nn_binary_classifier.
    """

    train_data = {'X': X, 'Y': Y, 'm': X.shape[1], 'n_x': X.shape[0], 'n_h': n_h}

    weights = initialize_weights(train_data['n_x'], n_h)

    loss_on_iteration = []

    for i in range(iterations):
        A2, cache = forward_propagation(X, weights)
        loss = loss_function(A2, train_data)
        loss_on_iteration.append(loss)
        gradients = back_propagation(A2, weights, train_data, cache)
        weights = update_weights(weights, gradients, learning_rate)
        if i % 250 == 0:
            print(f'Cost after {i} iteration: {loss_on_iteration[i]}')

    print(f'Final cost: {loss_on_iteration[-1]}')
    draw_loss(loss_on_iteration)
    return weights


def draw_loss(loss_on_iteration):
    iterations = len(loss_on_iteration)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(list(range(iterations)), loss_on_iteration)
    plt.show()


def predict_nn_binary_classifier(X: np.ndarray, weights: dict):
    """
    Prediction function for neural network.

    :param X: 2d array of data to predict, samples should be stored in columns.
    :param weights: dictionary of weights for nerual network, output of train_nn_binary_classifier.
    :return: Y_hat – row vector of probabilities that given sample belongs to class 1.
    """
    Y_hat, _ = forward_propagation(X, weights)
    return Y_hat

def normalize_data(X_train, X_test):
    """
    Normalizes data, each feature is scaled to have mean 0 and variance 1, based on train data.

    :param X_train: 2d array of training data, samples should be stored in columns.
    :param X_test: 2d array of test data, samples should be stored in columns.
    :return: X_train, X_test; normalized arrays
    """
    mi = np.mean(X_train, axis=1, keepdims=True)
    sigma = np.sqrt(np.mean(X_train**2, axis=1, keepdims=True))
    assert(mi.shape == (X_train.shape[0], 1))
    assert (sigma.shape == (X_train.shape[0], 1))
    X_train = (X_train - mi) / sigma
    X_test = (X_test - mi) / sigma
    return X_train, X_test
