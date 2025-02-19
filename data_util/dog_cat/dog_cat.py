import pandas as pd
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

image_dir = "~/ML/data_util/dog_cat/dog_vs_cat/cat_dog"
label_path = "~/ML/data_util/dog_cat/dog_vs_cat/cat_dog.csv"

"""
    In this file, I will show how to obtain the data from dag vs cat data set. The output data matrix will have dimension
    X in R^{d x n}
    y in R^{1 x n}
"""

"""
    For an image dataset, it is always useful to visualize the graph in order to get familiar
    The image is in jpg format, and python has PIL package to transform image data into vector   
"""


def visualize_graph(filename):
    image_path = os.path.join(image_dir, filename)
    img = Image.open(image_path).convert('RGB')
    plt.imshow(img)
    plt.show()


def load_data(samples=5000, train_test_ratio=0.7, rz=64):
    # create label dictionary
    label_csv = pd.read_csv(label_path)
    label_map = {}
    for i in range(len(label_csv["image"])):
        label_map[label_csv["image"][i]] = label_csv["labels"][i]

    # create the vectorized dataset
    vec_dataset = []
    labels = []
    image_files = os.listdir(image_dir)
    for filename in image_files[:samples]:
        # resize the image to fix the dimension
        img = Image.open(os.path.join(image_dir, filename)).convert("RGB").resize((rz, rz))
        vec_dataset.append((np.array(img) / 255).reshape(-1, 1))
        labels.append(label_map[filename])

    X_train = np.squeeze(np.array(vec_dataset[:int(samples * train_test_ratio)])).T
    y_train = np.array(labels[:int(samples * train_test_ratio)]).reshape(1, -1)
    X_test = np.squeeze(np.array(vec_dataset[int(samples * train_test_ratio):])).T
    y_test = np.array(labels[int(samples * train_test_ratio):]).reshape(1, -1)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
