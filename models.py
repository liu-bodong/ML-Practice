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