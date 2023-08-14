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
X = torch.rand(1,28,28, device = device)
logits = model(X)
pred_probab = nn.Softmax(dim = 1)(logits) # dim = 1 表示按行softmax
y_pred = pred_probab.argmax(1) # 返回每一行最大值的索引
print(f"Predicted class: {y_pred}")

### test the model on a batch of data ###











