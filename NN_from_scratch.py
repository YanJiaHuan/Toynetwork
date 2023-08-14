### lib ###
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

### device ###
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

### Class ###
class NuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() #用了super之后，就可以直接用nn.Module里的方法了
        self.flatten = nn.Flatten() # flatten 会把一个多维的tensor展平成一个一维的tensor
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512), # 写成28*28或784都可以
                nn.ReLU(), # 传统的激活函数
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
                nn.ReLU()
                )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

### init the model ###
model = NuralNetwork().to(device)

### use the model ###
# X = torch.rand(1,28,28, device = device)
# logits = model(X)
# pred_probab = nn.Softmax(dim = 1)(logits) # dim = 1 表示按行softmax
# y_pred = pred_probab.argmax(1) # 返回每一行最大值的索引
# print(f"Predicted class: {y_pred}")

### test the model on a batch of data ###
# x = torch.ones(5)
# y = torch.zeros(3)
# w = torch.randn(5,3, requires_grad = True)
# print(w.grad)
# b = torch.randn(3, requires_grad = True)
# z = torch.matmul(x,w) + b
# loss = nn.functional.binary_cross_entropy_with_logits(z,y)
# loss.backward()
# print(z.requires_grad)
# print(w.grad)

### training ###
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
        root = "data",
        train = True,
        download = True,
        transform = ToTensor()
        )
test_data = datasets.FashionMNIST(
        root = "data",
        train = False,
        download = True,
        transform = ToTensor()
        )
train_dataloader = DataLoader(training_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)
model = NuralNetwork().to(device)

## hyperparameters ##
learning_rate = 1e-3
batch_size = 64
epochs = 5
loss_fn = nn.CrossEntropyLoss()

## optimizer ##
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

## training loop ##
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # 这个的作用想当于 with torch.no_grad()=False 也就是说要更新参数
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred,y)

        # Backpropagation
        loss.backward()
        optimizer.step() # 这一步是更新参数
        optimizer.zero_grad() # 梯度清零,假如没有这一步，这次计算的梯度会和上次的梯度累加，这样的话会一直记着上次的梯度，损耗memory

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
## test loop ##
def test_loop(dataloader, model, loss_fn):
    model.eval() # 等于with torch.no_grad()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = loss_fn
optimizer = optimizer
epochs = epochs
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

### save and load the model ###










