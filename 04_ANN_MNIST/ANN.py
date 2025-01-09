import torch
from torch import nn as nn

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.hidden1 = nn.Linear(in_features=28*28, out_features=300)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(in_features=300, out_features=100)
        self.out = nn.Linear(in_features=100, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_x):
        x = self.hidden1(input_x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.softmax(x)
        return x
    
if __name__ == "__main__":
    
    X = torch.randn(28, 28, 1).reshape(1, 784)
    ann = ANN()
    print(ann)
    y = ann(X)
    print(y.shape)