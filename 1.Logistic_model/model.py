""" 
    In this package, a logistic model will be implemented from scratch.
    For all machine learning models, aligning the dimensionality of the data and the learnable parameters
    are important.

    For my model, What I require is:
    Input X: (features, num_of_samples)
    Label Y: (1, num_of_samples)
    Learnable parameters w: (features, 1)
    Bias b: scalar
"""

import numpy as np
from sympy.codegen.ast import float64

"""
    Sigmoid function takes in a scalar or an np array, and generate logistic value(s) of each element
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""
    initialize w and b to 0
"""


def initialize_to_zero(dim):
    b = float(0)
    return np.zeros((dim, 1)), b


"""
    Propagate pushes the data forward to generate 
        output (features, 1), 
        loss, 
        gradient
    For the formula derivation, see documents in the same directory
"""


def propagate(w, b, X_train, Y_train):
    # calculate sigmoid (1, num_of_samples)
    A = sigmoid(w.T @ X_train + b)
    # calculate the loss
    loss = -(np.dot(Y_train, np.log(A).T) + np.dot(1 - Y_train, np.log(1 - A).T)) / X_train.shape[1]
    # calculate dw and db
    dw = (X_train @ (A - Y_train).T) / X_train.shape[1]
    db = np.sum(A - Y_train) / X_train.shape[1]
    return loss, dw, db


"""
    optimize does the gradient descent
"""


def optimize(w, b, X_train, Y_train, runs=100, learning_rate=0.01):
    costs = []

    for i in range(runs + 1):
        loss, dw, db = propagate(w, b, X_train, Y_train)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            print(f'iteration: {i}, loss: {loss}')
            costs.append(loss)

    return w, b, costs


"""
    Model organize the functions together and do the training based on X_train and Y_train. 
    It outputs the learned parameters w and b. Also the loss in the end
"""


def Model(X_train, Y_train, runs=1000, learning_rate=0.01):
    w, b = initialize_to_zero(X_train.shape[0])

    w, b, costs = optimize(w, b, X_train, Y_train, runs, learning_rate)

    return w, b


"""
    Predict the label for given input Y_test, and see the accuracy of the predicted label
"""


def predict(w, b, X_test, y_test):
    Y_pred_sig = sigmoid(w.T @ X_test + b)
    Y_pred = np.zeros((1, y_test.shape[1]))
    for i in range(Y_pred_sig.shape[1]):
        if Y_pred_sig[0, i] > 0.5:
            Y_pred[0, i] = 1
        else:
            Y_pred[0, i] = 0

    acc = 1 - (len(Y_pred[Y_pred != y_test]) / y_test.shape[1])
    return acc
