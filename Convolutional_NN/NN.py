from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
import torch

device = torch.device('cpu')

'''
    In this file, we will start to learn an important tool: data loader, which helps to transform
    the dataset into iterable and to deal with a lot of things regarding to the dataset, 
    e.g, data shuffle, automatic batching, etc
    
    More details will be explained in ../data_loader
'''
train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST('./data', train=False, download=True, transform=ToTensor())

'''
    Batch size 
'''
BATCH_SIZE = 32

'''
    turn dataset into dataloader
    shuffle helps to make each batch contain a relatively evenly distributed labels
'''
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

'''
    check what is inside a dataloader
'''
train_features_batch, train_targets_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_targets_batch.shape)

'''
    Let's build a baseline model (an fully connected nn with 1 hidden layer)
    the nn.Flatten will flatten each data point in a batch, i.e, (C, H, W) -> (C, H * W)
'''

# hyperparameter
INPUT_DIM = 28 * 28
OUTPUT_DIM = len(train_data.classes)
layer_dims = [INPUT_DIM, 256, 128, 64, OUTPUT_DIM]
learning_rate = 0.001
epochs = 10 # How many times the entire dataset to be trained by the model
#dropout = 0.2

class FashionModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Flatten()]
        for i in range(1, len(layer_dims)-1):
            layers.append(torch.nn.Linear(layer_dims[i-1], layer_dims[i]))
            layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(layer_dims[-2], layer_dims[-1]))
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        self.layer_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer_stack(x)

model_0 = FashionModelV0()
model_0.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=learning_rate)

'''
    It is the time to train the model
    We want to track the training process and visualize it more clearly. 
    Here is some standard way to do it
    
    timer: used to record the time
    tqdm: a clear dynamic progress bar to track the current progress
    
    The minibatch training process is a little bit different from batch training process.
    
'''
from timeit import default_timer as timer
from tqdm.auto import tqdm

torch.manual_seed(42)
start = timer()

for epoch in tqdm(range(epochs)):

    print(f'Epoch {epoch}\n---------')

    # training
    train_loss = 0

    for batch, (X, y) in enumerate(train_dataloader):
        # set to train mode
        model_0.train()
        # forward
        train_pred = model_0(X.to(device))
        # loss, remember, y worked as index to access the value of the corresponding label of each data point in train_pred
        loss = loss_fn(train_pred, y.to(device))
        train_loss += loss.item()
        # zero out grad
        optimizer.zero_grad()
        # compute gradient
        loss.backward()
        # update gradient
        optimizer.step()
        # print out how many samples have been looked
        if batch % 200 == 0:
            print(f'Looked {BATCH_SIZE * batch}/{len(train_dataloader.dataset)} samples')

    # For each epoch, we average out the loss from each batch
    train_loss /= len(train_dataloader)

    # Within each epoch, we also want to test our model
    test_loss = 0
    test_acc = 0
    # stop recording the gradient
    with torch.no_grad():
        # switch to evaluation mode
        model_0.eval()
        for (X, y) in test_dataloader:
            # forward
            test_pred = model_0(X.to(device))
            # loss
            test_loss += loss_fn(test_pred, y.to(device)).item()
            # acc
            y_pred = test_pred.argmax(dim=1, keepdim=True).squeeze()
            test_acc += y_pred.eq(y.to(device)).sum().item()
        # average the test_loss
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader.dataset)

    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.5f}%\n")

end = timer()
total_time = end - start
print(f"Train time on {device}: {total_time:.3f} seconds")




