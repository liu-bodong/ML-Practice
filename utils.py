import torch
import torchvision
import models
from torch import nn
from torch.utils import data
from torchvision import transforms
# from matplotlib import pyplot as plt
# from d2l import torch as d2l

# from d2l import torch as d2l

class Accumulator:
    """Accumulator accumulates [n] variables.
    """
    def __init__(self, n):
        self.data = n * [0.0]
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
        

def load_data_fashion_mnist(batch_size, resize=None):
    """Load the dataset FashionMnist

    Args:
        batch_size (_type_): _description_
        resize (_type_, optional): _description_. Defaults to None.
    """
    transform = [transforms.ToTensor()]
    if resize:
        transform.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transform)
    
    train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test = torchvision.datasets.FashionMNIST(
        root='./data', train=False, transform=transform, download=True
    )
    
    return (data.DataLoader(train, batch_size, shuffle=True),
           data.DataLoader(test, batch_size, shuffle=False))
    
def accuracy(y_pred, y):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred.argmax(axis=1)
    cmp = y_pred.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(model, data_iter):
    if isinstance(model, torch.nn.Module):
        model.eval()
    metric = Accumulator(2) # count (1) num of accurate predictions and (2) total num of predictions
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(model(X), y), y.numel())
    return metric[0] / metric[1]

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

