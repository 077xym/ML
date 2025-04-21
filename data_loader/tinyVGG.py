import math

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn

'''
    This script shows how to use data augmentation
'''

'''
    Set device
'''
device = torch.device('cpu')

'''
    Obtaining the data
'''
train_path = './data/pizza_steak_sushi/train'
test_path = './data/pizza_steak_sushi/test'

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(31),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


'''
    remember, dataset object can be treated as a list of all data points, along with corresponding label
    
    You can also access all classes by .classes field
'''
train_dataset = ImageFolder(root=train_path, transform=train_transform)
test_dataset = ImageFolder(root=test_path, transform=test_transform)
class_names = train_dataset.classes

'''
    Convert dataset into data loader
    Remember: data loader is an iterator that splits and shuffle all your data points into mini batches, it also enables 
    many other features such as parallel loading (num_workers), Custom Sampling, etc
    
    dataloader can be considered as a list of all minibatches, where each batch contains BATCH_SIZE data points and labels
    However, one cannot access dataloader using indexing. The following are three ways of accessing data
    for Batch, (X, y) in enumerate(dataloader):
    for (X, y) in dataloader
    first_batch = next(iter(dataloader))
    
    you can also access .dataset field of dataloader, which will return the original dataset
'''
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=BATCH_SIZE)

'''
    Create CNN model
'''


def get_tensor_shape_from_conv(n_h, n_w, channels, k, s, p, pool_k, pool_s):
    for _ in range(len(channels) - 1):  # n channels -> n-1 layer.
        n_h = math.floor((n_h + 2 * p - k) / s + 1)
        n_w = math.floor((n_w + 2 * p - k) / s + 1)
        n_h = math.floor((n_h - pool_k) / pool_s + 1)
        n_w = math.floor((n_w - pool_k) / pool_s + 1)
    return channels[-1], n_h, n_w


# Hyperparameter
in_channel = train_dataset[0][0].shape[0]
layer_channels = [in_channel, 32, 64]
kernel_size = 3
kernel_padding = 0
kernel_stride = 1
pool_size = 2
pool_stride = 2
c, n_h, n_w = get_tensor_shape_from_conv(64, 64, layer_channels, kernel_size, kernel_stride, kernel_padding, pool_size,
                                         pool_stride)
layer_dims = [c * n_h * n_w, 128, len(class_names)]
learning_rate = 1e-5
dropout = 0.2
epochs = 15


class TinyVGG_0(nn.Module):

    def __init__(self):
        super().__init__()
        conv_layers = []
        for i in range(len(layer_channels) - 1):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=layer_channels[i],
                    out_channels=layer_channels[i + 1],
                    kernel_size=kernel_size,
                    stride=kernel_stride,
                    padding=kernel_padding,
                ))
            conv_layers.append(nn.ReLU())
            conv_layers.append(
                nn.MaxPool2d(
                    kernel_size=pool_size,
                    stride=pool_stride,
                ))
        self.conv_layer = nn.Sequential(*conv_layers)
        fc_layers = [nn.Flatten()]
        for i in range(len(layer_dims) - 2):
            fc_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        for l in fc_layers:
            if isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight, nonlinearity='relu')
                nn.init.zeros_(l.bias)
        self.fc_layer = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.fc_layer(self.conv_layer(x))


'''
    Test your model.
    
    Important! Pytorch CNN model expects input to be (batch_size, channel, height, width). So remember to add a dim 
    to align with the input dim requirement. 
'''

model_0 = TinyVGG_0()

# img, label = train_dataset[0]
# img = img.unsqueeze(0)
# print(img.shape)
# with torch.no_grad():
#     model_0.eval()
#     pred = model_0(img)
#
# pred_prob = torch.nn.functional.softmax(pred, dim=1)
#
# print(f'input shape: {img.shape}')
# print(f'output logits: {pred}')
# print(f'output prob: {pred_prob}')
# print(f'predicted label: {class_names[pred_prob.argmax(dim=1).item()]}')
# print(f'actual label: {class_names[label]}')


'''
    One other useful technique for dim check is torchinfo
'''
# from torchinfo import summary
#
# summary(model_0, input_size=(1, 3, 64, 64))

'''
    training process
'''

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=learning_rate)
torch.manual_seed(42)

from tqdm.auto import tqdm

for epoch in tqdm(range(epochs)):

    print(f'{epoch+1}/{epochs}\n---------')
    model_0.train()

    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        # forward
        train_pred = model_0(X)
        # calculate loss
        loss = loss_fn(train_pred, y)
        train_loss += loss.item()
        # zero out grad
        optimizer.zero_grad()
        # compute gradient
        loss.backward()
        # update grad
        optimizer.step()

        # visualize progress
        print(f'Looked at {(batch + 1) * BATCH_SIZE} samples')

    train_loss /= len(train_dataloader)

    with torch.no_grad():
        model_0.eval()
        test_loss = 0
        test_acc = 0
        for (X, y) in test_dataloader:
            # forward
            test_pred = model_0(X)
            # calculate loss
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            test_pred_label = torch.argmax(test_pred, dim=1).squeeze()
            test_acc += test_pred_label.eq(y).sum().item()

        test_loss /= len(test_dataloader)
        test_acc = test_acc / len(test_dataloader.dataset)

    print(f'train loss: {train_loss:.5f} | test loss: {test_loss:.5f} | test acc: {test_acc:.3f}%')


