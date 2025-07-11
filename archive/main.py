import utils
import models
import train
import torch
from torch import nn

if __name__ == "__main__":
    print("Use train")
    # batch_size = 256
    # num_epochs = 10
    # lr = 0.1
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # Load data
    # train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)

    # # Initialize model, loss function, and optimizer
    # model = models.MLP()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # # Train the model
    # test_acc = train.train(model, train_iter, test_iter, criterion, num_epochs, optimizer, device)
    
    # print(f'Final test accuracy: {test_acc:.3f}')