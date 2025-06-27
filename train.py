import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms

import utils
import models
import test

def train_epoch(model, train_iter, criterion, optimizer, device):
    model = model.to(device)
    if isinstance(model, nn.Module):
        model.train()
    # count (1) total training loss, (2) total training accuracy, and (3) num of samples
    train_loss, train_acc, num_samples = 0.0, 0.0, 0
    for X, y in train_iter:
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X)
        
        loss = criterion(y_pred, y)
        
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        
        with torch.no_grad():
            loss = loss.float()  
            train_loss += float(loss.sum())
            train_acc += test.accuracy(y_pred, y)
            num_samples += y.numel()

    return train_loss / num_samples, train_acc / num_samples

def train(model, train_iter, test_iter, criterion, num_epochs, optimizer, device):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(model, train_iter, criterion, optimizer, device)
        test_acc = test.evaluate_accuracy(model, test_iter, device)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, '
                  f'train loss {train_metrics[0]:.5f}, '
                  f'train acc {train_metrics[1] * 100:.3f} %, '
                  f'test acc {test_acc * 100:.3f} %')

    # evaluate the model
    test_acc = test.evaluate_accuracy(model, test_iter, device)
    return test_acc
     
    
if __name__ == "__main__":
    batch_size = 256
    num_epochs = 60
    lr = 0.1
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

    model = models.MLP()
    model.apply(utils.init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-2)

    print("Starting training...")
    test_accuracy = train(model, train_iter, test_iter, criterion, num_epochs, optimizer, device)
    print(f'Final test accuracy: {test_accuracy:.3f}')
   
    