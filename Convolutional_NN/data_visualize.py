import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

'''
    the torchvision package contains many useful datasets
    train: torchvision has split the data into train and test sets
    download: download if it does not exist in the assigned dir
    transform: transform PIL image data shape (HWC) into tensor (CHW)
'''

train_data = datasets.FashionMNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_data = datasets.FashionMNIST(root='./data', train=False, transform=ToTensor(), download=True)

'''
    The element in train_data has two parts: the actual data and its label
    The data is in pytorch tensor shape (CHW)
'''
image, label = train_data[0]
print(image.shape, label)

'''
    check the size of the data
    we can use .data to access all the data in the dataset
    we can use .target to access all the corresponding label of each data point
'''
print(len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets))

'''
    The .class can be used to access the class name in human language
'''
classes = train_data.classes
print(classes)

'''
    visualize the data
'''
# plt.imshow(image.squeeze(), cmap="gray")
# plt.axis(False)
# plt.title(classes[label])
# plt.show()

'''
    visualize multiple graphs
'''
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
row = col = 4
for i in range(1, row * col + 1):
    rand_idx = torch.randint(0, len(train_data), size=[1]).item()
    image, label = train_data[rand_idx]
    fig.add_subplot(row, col, i)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.axis(False)
    plt.title(classes[label])
fig.tight_layout()
plt.show()

