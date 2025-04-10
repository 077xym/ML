import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score

# Obtaining the data
data = pd.read_csv('Student_Performance.csv')
data["Extracurricular Activities"] = data["Extracurricular Activities"].map({'Yes': 1, 'No': 0})
train_split = int(len(data) * 0.8)
train_data = data.iloc[:train_split, :]
test_data = data.iloc[train_split:, :]

train_feature = torch.tensor(train_data.iloc[:, :-1].values.astype(np.float32))
train_label = torch.tensor(train_data.iloc[:, -1].values.astype(np.float32)).unsqueeze(dim=-1)

test_feature = torch.tensor(test_data.iloc[:, :-1].values.astype(np.float32))
test_label = torch.tensor(test_data.iloc[:, -1].values.astype(np.float32)).unsqueeze(dim=-1)

# Hyperparameters
runs = 5000
learning_rate = 0.002

# set a random seed to make model reproducible
torch.manual_seed(1337)


# model
class LinearRegression(nn.Module):

    def __init__(self):
        super().__init__()
        # linear regression can be thought of as a single layer neural network
        self.L1 = nn.Linear(train_feature.shape[1], 1, dtype=torch.float, bias=True)

    def forward(self, x):
        return self.L1(x)


model = LinearRegression()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training
for r in range(runs):
    # This step sets the model to train(default) mode
    # In train mode, dropout and batch norm(if there's any) will work
    model.train()
    # forward
    pred_label = model(train_feature)
    # compute loss
    loss = loss_fn(pred_label, train_label)
    # clear the grad as pytorch design grad computation as accumulated
    optimizer.zero_grad()
    # backward
    loss.backward()
    # grad update
    optimizer.step()
    # Set to evaluation mode
    # In this case, dropout will be disabled
    # batch norm will use the statistics from training data
    model.eval()
    if r % 100 == 0:
        with torch.inference_mode():
            pred_test_label = model(test_feature)
            test_loss = loss_fn(pred_test_label, test_label)

        print(f'run: {r} | train_loss: {loss.item()} | test_loss: {test_loss.item()}')

# performance
pred_test_label = model(test_feature)
r2 = r2_score(test_label.detach().numpy(), pred_test_label.detach().numpy())
print(r2)

# save the model parameters
torch.save(model.state_dict(), 'model.pth')
