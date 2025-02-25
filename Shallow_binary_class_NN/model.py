"""
    In this file, a shallow (with 1 hidden layer) neural network binary classifier will be implemented.
    The dimension of each matrix is shown below:
    X in R^{d x n}: input training dataset with n data points and d features, each column represents one data point.
    W1 in R^{n_l x d}: the weights matrix of hidden layer 1, which projects each data point into a vector with dim n_l
    b1 in R^{n_l x 1}: the bias for each neuron on the hidden layer 1
    W2 in R^{1 x n_l}: the weights matrix of output layer, which projects each neuron vector into a scalar
    b2 in R: the bias for the output scalar
    Y in R^{1 times n}: the vector that records the true label of each data point.

    The activation function for this hidden layer will be the tanh function

    And there will be a sigmoid function applying on the output scalar to predict the probability of the label
"""
import numpy as np

'''
    sigmoid function
    Use clip to prevent overflow, note that the sigmoid problem has most of its value change in between -6, 6, so 
    a reasonable clip range will not influence the model performance a lot
'''


def sigmoid(x):
    np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x))


'''
    The first step is to define the size of each matrix
    the output from this function are n_x, n_y and n_l, where
    W1 = n_l x n_x
    b1 = n_l x 1
    W2 = n_y x n_l
    b2 = n_y x 1
'''


def layer_size(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return n_x, n_y


'''
    Initialize the model parameters, and put all parameters into a table for a cleaner usage
    Note: when initializing weights matrices, we shall not initialize each parameter with same values. An issue called 
    model symmetry will occur if you do so. Check on the document for more detail.
    Note, do not directly return a dictionary, instead, always return a variable parameters, such there will be one 
    variable across the training process to record the current parameter states.
'''


def initialize_parameters(n_x, n_y, n_l):
    # Xavier initialization for numerical stability
    W1 = np.random.randn(n_l, n_x) * np.sqrt(1 / n_x)
    b1 = np.zeros((n_l, 1))
    W2 = np.random.randn(n_y, n_l) * np.sqrt(1 / n_x)
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


'''
    Propagation: How training data transform when sent to the model
    Z1 = W1 @ X + b1 in R^{n_l x n}
    A1 = sigma(Z1) in R^{n_l x n}. The sigma will be tanh in this document. TODO: make it general
    Z2 = W2 @ A1 + b2 in R^{1 x n}
    A2 = sigmoid(Z2) in R^{1 x n}
    What this function will return are the final output A2, and a cache that stores Z1, A1, Z2 and A2 for the usage of 
    gradient computation
    
    Also, to avoid log0 run time error, clip the output within the range epsilon - 1-epsilon
'''


def propagate(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = W1 @ X + b1
    A1 = np.tanh(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    epsilon = 1e-8
    np.clip(A2, epsilon, 1 - epsilon)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache


'''
    Compute the loss for the entire training dataset. This is mainly used to track how loss changes and thus tune the 
    parameters for the model.
'''


def compute_loss(A2, Y):
    n = Y.shape[1]
    cost = -(np.dot(np.log(A2), Y.T) + np.dot(np.log(1 - A2), (1 - Y).T)) / n
    cost = float(np.squeeze(cost))
    return cost


'''
    Back-propagation. This method use back-propagation to compute the gradient of the loss w.r.t each weights
    For a detailed analysis on how the gradients are computed, please check the document at 
    ../Underlying_methods/Gradient_computation.pdf section 4.
'''


def back_propagate(X, Y, parameters, cache):
    n = Y.shape[1]

    ## after you have coded the gradients, you can delete parameters that are not used
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]

    dZ2 = (A2 - Y) / n
    dW2 = dZ2 @ A1.T
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = ((W2.T @ dZ2) * (1 - np.power(A1, 2))) / n
    dW1 = dZ1 @ X.T
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


'''
    Now, update the gradient of the loss w.r.t each weights by learning rate. 
    This method will be used in the gradient descent loop.
'''


def update_gradient(parameters, grads, learning_rate=0.1):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    return parameters


'''
    put everything together to form a model
'''


def shallow_nn_model(X, Y, n_l, learning_rate=0.1, runs=1000):
    n_x, n_y = layer_size(X, Y)

    parameters = initialize_parameters(n_x, n_y, n_l)

    for i in range(runs):

        A2, cache = propagate(X, parameters)

        grads = back_propagate(X, Y, parameters, cache)

        parameters = update_gradient(parameters, grads, learning_rate)

        if i % 100 == 0:
            loss = compute_loss(A2, Y)
            print(f'Current iteration: {i}; loss: {loss}')

    return parameters


'''
    Predict the result from a trained model, and return the accuracy
'''


def predict(X, Y, parameters):
    y_pred, cache = propagate(X, parameters)

    y_pred = np.where(y_pred > 0.5, 1, 0)

    return 1 - len(y_pred[y_pred != Y]) / Y.shape[1]
