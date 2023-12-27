# Author: Aditya
# Date: 2023-07-07

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from utils.dataloader import EEGDataset

import numpy as np
import pandas as pd

class trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, epochs: int, lr: float, device: str="cpu"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = lr
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        self.train_acc = []
        self.val_acc = []
        self.test_acc = []
    
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            loss_epoch = 0
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                # Get the data to cuda
                data = data.to(device=self.device).float()
                targets = targets.to(device=self.device)

                # forward
                scores = self.model(data)
                loss = self.criterion(scores, targets)
                loss_epoch += loss.item()

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                self.optimizer.step()
            
            # Calculate loss and accuracy for epoch
            self.train_losses.append(loss_epoch/len(self.train_loader))
            self.train_acc.append(self.eval(self.train_loader))
            self.val_losses.append(self.eval(self.val_loader))
            self.val_acc.append(self.eval(self.val_loader))

            # Print the loss for every 20th epoch
            if epoch % 20 == 0:
                print(f"Epoch: {epoch}, Train Loss: {self.train_losses[-1]}, Val Loss: {self.val_losses[-1]}")
    
    def eval(self, loader: DataLoader):
        self.model.eval()
        num_correct = 0
        num_samples = 0
        loss_epoch = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device).float()
                y = y.to(device=self.device)

                scores = self.model(x)
                loss = self.criterion(scores, y)
                loss_epoch += loss.item()

                _, predictions = scores.max(1)
                _, actual = y.max(1)
                num_correct += (predictions == actual).sum()
                num_samples += predictions.size(0)
        
        return loss_epoch/len(loader), float(num_correct)/float(num_samples)*100
