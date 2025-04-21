import math

import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
from timeit import default_timer as timer

'''
    Based on my experience, mps does not have a good performance. Therefore I will continue use cpu. Since cpu is 
    the default device, no need to transform tensor and models explicitly to cpu
'''
device = torch.device("cpu")

'''
    From NN.py, with a tuned hyperparameter set, we obtain a ~88% accuracy on testing data, which is good, but can we do better?
    Yes!, Using CNN to reach ~92% (without crop, zooming, etc)
'''

'''
    same steps to create data loader
'''

train_data = datasets.FashionMNIST(root="./data", train=True, transform=ToTensor(), download=True)
test_data = datasets.FashionMNIST(root="./data", train=False, transform=ToTensor(), download=True)

BATCH_SIZE = 64

'''
    Remember, dataloader is an iterator, you can:
    1. access all data points using .dataset. .dataset is a list containing each (X, y)
    2. use next(iter(dataloader)) to access the next batch in data loader, which in our case is 32 data points and 32 classes
    3. for batch, (X, y) in enumerate(dataloader) can access each batch in dataloader in a loop
    4. Note, indexing is not supported in dataloader
'''

train_dataset = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

'''
    Conv NN contains conv layer, pooling layer and a fully connected NN layer.
    The input dataset should be in the shape (B, C, H, W)
    In pytorch, Conv layer is made simple by nn.Conv2d, where you only need to specify
    in_channels: how many channels the input data has
    out_channels: the desired output channel (usually becomes larger)
    kernel_size: the size of the square rolling on the tensor
    padding: pad original tensor to make each matrix in the output tensor has the same size as input
    stride: steps the kernel(filter) moves
'''

'''
    Sometimes the math may become complicated and error-prone when calculating the dimension after conv layer. To resolve
    this issue, we can either
    1. Create a helper function to calculate final tensor shape
    2. Create separate module and use nn.Flatten to automatically fit to nn layer

    To make the script looks simple and tuning process smooth, I will choose the first method
'''


def get_tensor_shape_from_conv(n_h, n_w, channels, k, s, p, pool_k, pool_s):
    for _ in range(len(channels) - 1):  # n channels -> n-1 layer.
        n_h = math.floor((n_h + 2 * p - k) / s + 1)
        n_w = math.floor((n_w + 2 * p - k) / s + 1)
        n_h = math.floor((n_h - pool_k) / pool_s + 1)
        n_w = math.floor((n_w - pool_k) / pool_s + 1)
    return channels[-1], n_h, n_w


# conv layer Hyperparameter
IN_CHANNEL = train_dataset.dataset[0][0].shape[0]
layer_channels = [IN_CHANNEL, 32, 64]
kernel = 3
stride = 1
padding = 1
pooling_kernel = 2
pooling_stride = 2
# fcnn Hyperparameter
ch, n_h, n_w = get_tensor_shape_from_conv(train_dataset.dataset[0][0].shape[1],
                                          train_dataset.dataset[0][0].shape[2],
                                          layer_channels, kernel, stride, padding,
                                          pooling_kernel, pooling_stride)
layer_dims = [ch * n_h * n_w, 128, len(train_data.classes)]
learning_rate = 0.003
dropout = 0.5
epochs = 20


# Conv layer
class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        # conv layers
        layers = []
        for i in range(len(layer_channels) - 1):
            layers.append(nn.Conv2d(
                in_channels=layer_channels[i],
                out_channels=layer_channels[i + 1],
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            ))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=pooling_kernel, stride=pooling_stride))
        self.conv_layers = nn.Sequential(*layers)
        # ffwd layers
        ffwd_layers = [nn.Flatten()]
        for i in range(len(layer_dims) - 2):
            ffwd_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            ffwd_layers.append(nn.ReLU())
            ffwd_layers.append(nn.Dropout(dropout))
        ffwd_layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        for l in ffwd_layers:
            if isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight, nonlinearity='relu')
                nn.init.zeros_(l.bias)
        self.ffwd_layers = nn.Sequential(*ffwd_layers)

    def forward(self, x):
        return self.ffwd_layers(self.conv_layers(x))


conv_model = ConvNet()

'''
    Test the model layout by input a random batch of size (32, 1, 28, 28) to the model
    expected: (32, 1, 28, 28) -> (32, 10)
    This code snippet shall not appear in the final script, comment it out after your layout has been checked correctly
'''
# x = torch.rand((32, 1, 28, 28))
# output = conv_model(x)
# print(output.shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(conv_model.parameters(), lr=learning_rate)
torch.manual_seed(42)

s = timer()
for epoch in tqdm(range(epochs)):

    print(f'Epoch {epoch + 1}/{epochs}\n-------')

    conv_model.train()

    train_loss = 0
    for batch, (X, y) in enumerate(train_dataset):
        # forward
        train_pred = conv_model(X)
        # compute loss
        loss = loss_fn(train_pred, y)
        train_loss += loss.item()
        # zero out grad
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update
        optimizer.step()

        # visualize
        if batch % 200 == 0:
            print(f'looked Batch {batch}/{len(train_dataset)} samples')

    # batch-wisely average train_loss
    train_loss /= len(train_dataset)

    # test
    with torch.no_grad():
        conv_model.eval()
        test_loss = 0
        test_acc = 0
        for X, y in test_dataset:
            # forward
            test_pred = conv_model(X)
            # loss
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            # accuracy (all correct prediction from one batch)
            test_acc += test_pred.argmax(dim=1).squeeze().eq(y).sum().item()

    # batch-wise average loss
    test_loss /= len(test_dataset)
    # accuracy over entire test dataset
    test_acc /= len(test_dataset.dataset)

    print(f'Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.5f}%')

e = timer()
total_time = e - s
print(f"Train time on {device}: {total_time:.3f} seconds")

'''
    One great visualization for the performance metric is confusion matrix. Where we create a matrix A of classes by classes
    A[i][j] represents how many times the actual label i is predicted to be lable j
'''

y_preds = []
with torch.no_grad():
    conv_model.eval()
    for X, y in test_dataset:
        # forward
        test_pred = conv_model(X)
        # get y_pred
        y_pred = torch.softmax(test_pred, dim=1).argmax(dim=1)
        # append
        y_preds.append(y_pred)

y_pred_tensor = torch.cat(y_preds, dim=0)

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(test_data.classes), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
    class_names=test_data.classes,  # turn the row and column labels into class names
    figsize=(10, 7)
)
