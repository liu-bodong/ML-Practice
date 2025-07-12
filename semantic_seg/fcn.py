import torch
import torchvision
from torch import nn
from torch.nn import functional as F 
from collections import OrderedDict
from d2l import torch as d2l

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
def load_resnet():
    pretrained_net = torchvision.models.resnet18(pretrained=True)   
    return nn.Sequential(*list(pretrained_net.children())[:-2])
    
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

def load_data_voc(batch_size, crop_size, voc_dir=None):
    """Load the VOC semantic segmentation dataset.

    Defined in :numref:`sec_semantic_segmentation`"""
    if voc_dir is None:
        # Check if dataset already exists locally
        local_voc_dir = os.path.join('VOCdevkit', 'VOC2012')
        if os.path.exists(local_voc_dir):
            print("Use local data")
            voc_dir = local_voc_dir
        else:
            print("Download data")
            voc_dir = d2l.download_extract('voc2012', os.path.join(
                'VOCdevkit', 'VOC2012'))
    
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        d2l.VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        d2l.VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter

if __name__ == "__main__":    
    pretrained_resnet = torchvision.models.resnet18(weights=True)
    trimmed_resnet = list(pretrained_resnet.children())[:-2]
    print("training")
    
    num_class = 21
    
    W = bilinear_kernel(num_class, num_class, 64)
    convT = nn.ConvTranspose2d(num_class, num_class, kernel_size=64, padding=16, stride=32)
    convT.weight.data.copy_(W)
    
    net = nn.Sequential(*trimmed_resnet,
                        nn.Conv2d(512, num_class, 1),
                        convT)

    voc_dir = os.path.relpath("data\VOCdevkit\VOC2012")
    batch_size, crop_size = 32, (320, 480)
    train_iter, test_iter = load_data_voc(batch_size, crop_size, voc_dir)   
    
    num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
    
        
    
    