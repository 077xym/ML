import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

# device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# set a random seed for reproducibility
torch.manual_seed(1337)

# load data
mnist = fetch_openml('mnist_784', version=1)

X = mnist.data.astype('float32').values
y = mnist.target.astype('int64').values

X /= 255

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.from_numpy(X_train).to(device)  # (56000, 784)
X_test = torch.from_numpy(X_test).to(device)  # (14000, 784)
y_train = torch.from_numpy(y_train).to(device)  # (56000, )
y_test = torch.from_numpy(y_test).to(device)  # (14000, )

num_classes = 10
num_features = 784

# Hyperparameter
layer_dims = [num_features, 256, 128, 64, num_classes]
dropout = 0.2
learning_rate = 0.0003
runs = 1200


# Model
class MultiClassClassifier(torch.nn.Module):

    def __init__(self):
        super().__init__()
        layers = []
        for i in range(1, len(layer_dims) - 1):
            layers.append(torch.nn.Linear(layer_dims[i - 1], layer_dims[i], bias=True, device=device))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(layer_dims[-2], layer_dims[-1], bias=True, device=device))
        for layer in layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                torch.nn.init.zeros_(layer.bias)
        self.ffwd = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.ffwd(x)


# Training
model = MultiClassClassifier()
model.to(device)
# Note, this will automatically add a softmax layer and calculate the entropy
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for r in range(runs):

    # set to train mode
    model.train()
    # forward
    y_train_pred = model(X_train)
    # compute loss
    loss = loss_fn(y_train_pred, y_train)
    # clear the grad as pytorch design grad computation as accumulated
    optimizer.zero_grad()
    # backwards
    loss.backward()
    # update
    optimizer.step()

    # eval
    if r % 100 == 0:
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            loss_test = loss_fn(y_test_pred, y_test)

        print(f'runs:{r} | train loss: {loss.item()} | test loss: {loss_test.item()}')

# Performance
y_train_pred = model(X_train)
y_train_pred_prob = F.softmax(y_train_pred, dim=-1)
y_train_pred_label = torch.argmax(y_train_pred_prob, dim=-1)
correct_train = torch.eq(y_train_pred_label, y_train).sum().item()

y_test_pred = model(X_test)
y_test_pred_prob = F.softmax(y_test_pred, dim=-1)
y_test_pred_label = torch.argmax(y_test_pred_prob, dim=-1)
correct_test = torch.eq(y_test_pred_label, y_test).sum().item()

acc_train = correct_train / len(y_train)
acc_test = correct_test / len(y_test)
print(y_train_pred_label[:50], y_train[:50])
print(y_test_pred_label[:50], y_test[:50])
print(f'Train Accuracy: {acc_train:.4f}% | Test Accuracy: {acc_test:.4f}%')

# save
torch.save(model.state_dict(), './model.pth')
