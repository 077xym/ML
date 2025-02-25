"""
    This file contains the implementation of a deep neural network model with variable layer length L
    Suppose the activation function is relu, and the output layer is sigmoid
    TODO, make the activation function for both hidden and output layer general instead of hard coded for this model
"""
import copy
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


"""
    This function computes the gradient of the sigmoid function
"""


def backward_sigmoid(x):
    x = sigmoid(x)
    return x * (1 - x)

"""
    This function computes the gradient of the relu function
"""


def backward_relu(x):
    x_grad = np.where(x > 0, 1, 0)
    return x_grad


"""
    We first initialize the weights. The input layers_dims contains the dimensionality of each layer. The first element
    will be the dimensionality of the feature vector of the input. For example, if the input X has dimension R^{5 x n}, 
    the first hidden layer has 7 neurons, the third hidden layer has 3 neurons, and the last hidden layer (output layer)
    has 1 neuron, then, layers_dims will be [5, 7, 3, 1]. We need to initialize 3 weight matrices and 3 biases. The 
    weight matrices each has dimension 7 by 5, 3 by 7 and 1 by 3, the bias each has dimension 7, 3, 1.
    We will store all the weights and biases into a parameters. The complicated part is to figure out the indexing.
"""


def initialize_weights(layers_dims):
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters[f'W{l}'] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layers_dims[l], 1))
    return parameters


"""
    Then it is time to propagate the input X. We basically need to do 2 things here:
    1. generate the final output AL
    2. store the intermediate hidden output of each layer for the usage of back-propagation
    For each layer l, we need to store A^(l-1), W^(l) and Z^(l)
    You will get to know why these are needed later in back-propagation.
    
    Again the indexing is a little bit of complicated. The basic line is to regard l as the l^th layer
"""


def forward_propagate(X, parameters):
    # parameters contain weights and biases of each layer. Thus has 2L of length
    L = len(parameters) // 2
    # caches stores all the required parameters and outputs.
    caches = []
    A_prev = X

    # Since the hidden activation and the output activation are different, we need to run the fist L-1 layers
    for l in range(1, L):
        Wl = parameters[f'W{l}']
        bl = parameters[f'b{l}']
        Zl = Wl @ A_prev + bl
        Al = relu(Zl)

        caches.append((A_prev, Wl, Zl))
        A_prev = Al

    # Now propagate to the output layer
    WL = parameters[f'W{L}']
    bL = parameters[f'b{L}']
    ZL = WL @ A_prev + bL
    AL = sigmoid(ZL)

    caches.append((A_prev, WL, ZL))

    return AL, caches


"""
    compute_loss will compute the loss of the network in current state. Since it is a binary classifier, the loss will
    be cross-entropy loss
"""


def compute_loss(AL, Y):
    n = Y.shape[1]
    cost = -(np.dot(np.log(AL), Y.T) + np.dot(np.log(1 - AL), (1 - Y).T)) / n
    return np.squeeze(cost)


"""
    Here is the most complicated part, the back-propagation. For a detailed explanation on the equations, please refer
    to the document talking about gradient computation under the Underlying_methods directory. Here I will list the equations
    and you can take it as for granted.
    On layer l, we have:
    - dZl = dAl * g'(Zl) where dAl = W^{l+1}.T @ dZ^{l+1}
    - dWl = dZl @ A^{l-1}.T
    - dbl = dZl @ 1
    We can see that dAl depends on dZ^{l+1}, meaning that dZl depends on dZ^{l+1}, which leads to the kernel of 
    back-propagation.
    
    To implement this, we see that at layer l+1, we can get dAl, which depends on the cache of W^{l+1}, and dZ^{l+1}. Thus,
    when we are going backwards, for a certain layer l, what we need is to compute
    - dZl shown above
    - dWl shown above
    - dbl shown above
    - dA^{l-1} = Wl.T @ dZl
    
    Thus, in forward propagation, we need to store A^{l-1}, Wl and Zl for each layer l in order to compute the gradient
    
    The base cases dAL, dZL, dWL and dbL will be computed separately.
    - dAL = -1/n (np.divide(Y, AL)-np.divide(1-Y, 1-AL))
"""


def backward_propagate(AL, Y, caches):
    n = Y.shape[1]
    grads = {}
    # remember, the element in the cache is a tuple containing A^{l-1}, Wl and Zl for layer l. So the length of caches
    # is the number of layers
    L = len(caches)

    # compute the gradient of last layer
    cache_L = caches[L - 1]
    A_prev = cache_L[0]
    WL = cache_L[1]
    ZL = cache_L[2]
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) / n
    dZL = dAL * backward_sigmoid(ZL)
    dWL = dZL @ A_prev.T
    dbL = np.sum(dZL, axis=1, keepdims=True)
    dA_prev = WL.T @ dZL
    grads[f'dW{L}'] = dWL
    grads[f'db{L}'] = dbL
    grads[f'dA{L-1}'] = dA_prev

    # back propagate starting from L - 1 layer. Note that in this case, l + 1 is actually the current layer.
    for l in reversed(range(L - 1)):
        cache_current = caches[l]
        A_prev_current = cache_current[0]
        W_current = cache_current[1]
        Z_current = cache_current[2]

        dZ_current = grads[f'dA{l+1}'] * backward_relu(Z_current)
        dW_current = dZ_current @ A_prev_current.T
        db_current = np.sum(dZ_current, axis=1, keepdims=True)
        dA_prev = W_current.T @ dZ_current

        grads[f'dW{l+1}'] = dW_current
        grads[f'db{l+1}'] = db_current
        grads[f'dA{l}'] = dA_prev

    return grads


"""
    After we compute the gradient, we can now begin to update the parameters
"""


def update_parameters(parameters, grads, learning_rate=0.1):
    L = len(parameters) // 2
    parameters = copy.deepcopy(parameters)

    for l in range(1, L+1):
        parameters[f'W{l}'] = parameters[f'W{l}'] - learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] = parameters[f'b{l}'] - learning_rate * grads[f'db{l}']

    return parameters


"""
    Use the trained model the predict the X_test, and use the prediction accuracy as the model performance metric
"""


def predict(X_test, Y_test, parameters):
    Y_pred, _ = forward_propagate(X_test, parameters)
    Y_pred = np.where(Y_pred > 0.5, 1, 0)
    return 1 - len(Y_pred[Y_pred != Y_test]) / Y_test.shape[1]


"""
    Now we can put all methods together
"""


def DNN(X_train, Y_train, layers_dims, iterations=1000, learning_rate=1.0):
    # initialize the parameters
    parameters = initialize_weights(layers_dims)

    for r in range(iterations):

        # forward propagate
        AL, caches = forward_propagate(X_train, parameters)

        # backward propagate
        grads = backward_propagate(AL, Y_train, caches)

        # update gradient
        parameters = update_parameters(parameters, grads, learning_rate)

        if r % 100 == 0:
            loss = compute_loss(AL, Y_train)
            print(f'current interation: {r}, Loss: {loss}')

    return parameters


if __name__ == '__main__':
    pass