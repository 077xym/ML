from Shallow_binary_class_NN.model import shallow_nn_model, predict
from data_util.dog_cat.dog_cat import load_data

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data(samples=2000, train_test_ratio=0.7, rz=32)
    n_l = X_train.shape[0]
    parameters = shallow_nn_model(X_train, y_train, n_l, runs=3000, learning_rate=1.5)
    y_pred_train_acc = predict(X_train, y_train, parameters)
    y_pred_test_acc = predict(X_test, y_test, parameters)
    print(f'train_set_acc:{y_pred_train_acc}\ntest_set_acc:{y_pred_test_acc}')
