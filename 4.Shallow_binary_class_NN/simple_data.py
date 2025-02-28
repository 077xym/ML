from Shallow_binary_class_NN.model import shallow_nn_model, predict
from planar_utils import load_planar_dataset

if __name__ == '__main__':
    X, Y = load_planar_dataset()

    parameters = shallow_nn_model(X, Y, 6, learning_rate=1.2, runs=10000)

    acc = predict(X, Y, parameters)
    print(acc)