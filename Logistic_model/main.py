from Logistic_model.model import Model, predict
from data_util.dog_cat.dog_cat import load_data

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    w, b = Model(X_train, y_train, runs=8000, learning_rate=0.003)
    acc1 = predict(w, b, X_test, y_test)
    acc2 = predict(w, b, X_train, y_train)
    print(f'test accuracy is {acc1}, train accuracy is {acc2}')
