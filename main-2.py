from utils.dataloader import EEGDataset
from models.cnn import testEEGNet, EEGNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Set the random seed
    torch.manual_seed(42)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the hyperparameters
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 1000

    # Create the dataset
    dataset_path = "/media/data/PRECOG_Data/2022N400_Epoched/"
    train_subjects = ["sub-{:02d}".format(i) for i in range(1, 20) if i not in [5, 10, 15, 18]]
    train_dataset = EEGDataset(dataset_path, subjects=train_subjects)


    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create the test dataloader
    test_subjects = ["sub-{:02d}".format(i) for i in range(20, 24) if i not in [5, 10, 15, 18]]
    test_dataset = EEGDataset(dataset_path, subjects=test_subjects)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Create the model
    model = EEGNet().to(device)

    # Initialize the weights
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)
    model.apply(init_weights)

    # Create the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    losses = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        loss_epoch = 0
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Get the data to cuda
            data = data.to(device=device).float()  

            # Convert the targets to longs
            targets = targets.to(device=device).float()

            # forward
            scores = model(data)

            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            loss_epoch += loss.item()

        losses.append(loss_epoch)
        print(f"Cost at epoch {epoch} is {loss_epoch}")

        # Test the model
        def check_accuracy(loader, model):
            model.eval()
            num_correct = 0
            num_samples = 0
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device=device).float()
                    y = y.to(device=device).float()

                    scores = model(x)
                    _, predictions = scores.max(1)
                    _, actual = y.max(1)

                    num_correct += (predictions == actual).sum()
                    num_samples += predictions.size(0)

                print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
                return float(num_correct)/float(num_samples)*100
        
        if epoch % 10 == 0:
            check_accuracy(dataloader, model)
            val_acc = check_accuracy(test_dataloader, model)
            val_accs.append(val_acc)
            print("Max val acc: ", max(val_accs))
        
        # Save the model with the best validation accuracy
        if val_acc == max(val_accs):
            torch.save(model.state_dict(), "checkpoints/test_eegnet.pth")


        
