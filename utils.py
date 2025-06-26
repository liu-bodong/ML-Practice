import torch
import torchvision
from models import Network
from torch import nn
from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt
from d2l import torch as d2l

# from d2l import torch as d2l

class Accumulator:
    """Accumulator accumulates [n] variables.
    """
    def __init__(self, n):
        self.data = n * [0.0]
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self,data, args)]
    
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

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2) # count (1) num of accurate predictions and (2) total num of predictions
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch(net, train_iter, criterion, optimizer, device):
    net = Network().to
    if isinstance(net, nn.Module):
        net.train()
    # count (1) total training loss, (2) total training accuracy, and (3) num of samples
    metric = Accumulator(3)
    for X, y in train_iter:
        y_pred = net(X)
        loss = criterion(y_pred, y)
        
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        
        metric.add(float(loss.sum()), accuracy(y_pred, y), y.numel())
        
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_iter, test_iter, criterion, num_epochs, optimizer):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, criterion, optimizer)
        test_acc = evaluate_accuracy(net, test_iter)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, '
                  f'train loss {train_metrics[0]:.3f}, '
                  f'train acc {train_metrics[1]:.3f}, '
                  f'test acc {test_acc:.3f}')
    
    #draw the training curve
    
