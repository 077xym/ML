import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import pandas as pd

from Logistic_model.model import Model, predict

"""
    In this file, I will show how to train the model and save the result. 
    
    One of the most important steps is to process the data that fits the model. However, different dataset has
    different formats, so the processing step will vary. 
    
    I will train a data set containing a total of 25000 pictures of cats and dogs. Along with a label 0(cat) or 1(dog) to each graph
    
    Although not all datasets have same format as this dataset, in this file, some methods that I used are general and are
    worth to know.
"""

"""
    For an image dataset, it is always useful to visualize the graph in order to get familiar
    The image is in jpg format, and python has PIL package to transform image data into vector   
"""

image_dir = "./dog_vs_cat/cat_dog"
label_path = "./dog_vs_cat/cat_dog.csv"


def visualize_graph(filename):
    image_path = os.path.join(image_dir, filename)
    img = Image.open(image_path).convert('RGB')
    plt.imshow(img)
    plt.show()


def load_data():
    # create label dictionary
    label_csv = pd.read_csv(label_path)
    label_map = {}
    for i in range(len(label_csv["image"])):
        label_map[label_csv["image"][i]] = label_csv["labels"][i]

    # create the vectorized dataset
    vec_dataset = []
    labels = []
    samples = 5000
    train_test_ratio = 0.7
    image_files = os.listdir(image_dir)
    for filename in image_files[:samples]:
        # resize the image to fix the dimension
        img = Image.open(os.path.join(image_dir, filename)).convert("RGB").resize((64, 64))
        vec_dataset.append((np.array(img)/255).reshape(-1, 1))
        labels.append(label_map[filename])

    X_train = np.squeeze(np.array(vec_dataset[:int(samples*train_test_ratio)])).T
    y_train = np.array(labels[:int(samples*train_test_ratio)]).reshape(1, -1)
    X_test =np.squeeze(np.array(vec_dataset[int(samples*train_test_ratio):])).T
    y_test = np.array(labels[int(samples*train_test_ratio):]).reshape(1, -1)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    w, b = Model(X_train, y_train, runs=15000, learning_rate=0.005)
    acc1 = predict(w, b, X_test, y_test)
    acc2 = predict(w, b, X_train, y_train)
    print(f'test accuracy is {acc1}, train accuracy is {acc2}')
