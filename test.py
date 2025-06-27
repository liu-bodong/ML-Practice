import torch
import torchvision
import models
from torch import nn
from torch.utils import data
from torchvision import transforms

    
def accuracy(y_pred, y):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred.argmax(axis=1)
    cmp = y_pred.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(model, data_iter, device):
    if isinstance(model, torch.nn.Module):
        model.eval()
    acc_preds, total_preds = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            acc_preds += accuracy(model(X), y)
            total_preds += y.numel()
    return acc_preds / total_preds


