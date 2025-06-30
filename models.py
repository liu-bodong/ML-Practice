import torch 
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten the input
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        return torch.log_softmax(self.layer3(x), dim=1)
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop1 = nn.Dropout2d()
        self.fc1   = nn.Linear(320, 50)
        self.fc2   = nn.Linear(50, 10)
        self.bn    = nn.BatchNorm2d(20)
        
    def forward(self, x):
        x = nn.max_pool2d(self.conv1(x), 2)
        x = nn.relu(x) + nn.relu(-x)
        x = nn.relu(nn.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        x = x.view(-1, 320)
        x = nn.relu(self.fc1(x))
        x = nn.drop1(x, training=self.training)
        x = self.fc2(x)
        x = nn.softmax(x, dim=1)
        return x