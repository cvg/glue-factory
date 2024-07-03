import os
import numpy as np

import torch
from torch.utils.data import *
from torch.optim import *
from torch import nn

import matplotlib.pyplot as plt
from sklearn import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLPDataset(Dataset):
    def __init__(self, isTrain=True):
        if isTrain:
            self.positives = torch.from_numpy(np.load('mlp_data/positives.npy')).float()
            self.negatives = torch.from_numpy(np.load('mlp_data/negatives.npy')).float()
        else:
            self.positives = torch.from_numpy(np.load('mlp_data/positives_test.npy')).float()
            self.negatives = torch.from_numpy(np.load('mlp_data/negatives_test.npy')).float()

        self.entries = torch.concatenate((self.positives, self.negatives), axis=0).to(device)
        self.labels = torch.concatenate((torch.ones(self.positives.shape[0]), torch.zeros(self.negatives.shape[0])), axis=0).to(device)

    def __getitem__(self, idx):
        return self.entries[idx], self.labels[idx]

    def __len__(self):
        return len(self.entries)


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(150, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 32)
        self.layer5 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.ReLU()(x)
        x = self.layer2(x)
        x = torch.nn.ReLU()(x)
        x = self.layer3(x)
        x = torch.nn.ReLU()(x)
        x = self.layer4(x)
        x = torch.nn.ReLU()(x)
        x = self.layer5(x)
        return torch.sigmoid(x)

def train(model, train_data, val_data):
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    for _ in torch.arange(1, 5):
        train_loss = 0.0

        for _, batch in enumerate(train_data):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            # Reset the gradient
            optimizer.zero_grad()

            # Forward pass through the model
            x_pred = model(x)

            # Compute the loss
            loss = nn.BCELoss()(x_pred.reshape(-1), y.float())

            # Backpropagation
            loss.backward()

            # Update the model parameters
            optimizer.step()

            # Add to loss
            train_loss += loss.item()

        train_loss /= len(train_data)

        validation_loss = 0.0

        # print validation loss
        for _, batch in enumerate(val_data):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            # Reset the gradient
            optimizer.zero_grad()

            with torch.no_grad():

                # Forward pass through the model
                x_pred = model(x)

                # Compute the loss
                loss = nn.BCELoss()(x_pred.reshape(-1), y.float())

                # Add to loss
                validation_loss += loss.item()

        print(f"Train : {train_loss:.4f} Validation : {validation_loss / len(val_data):.4f}")

    # Save the final model for inference
    torch.save(model.state_dict(), f"mlp_data/mlp.pth")

    return model

if __name__ == '__main__':
    train_dataset = MLPDataset(isTrain=True)
    test_dataset = MLPDataset(isTrain=False)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    validation_dataloder = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

    model = MLPModel().to(device)

    final_model = train(model, train_dataloader, validation_dataloder)

    # Plot confusion matrix
    actual = []
    predicted = []

    for _, batch in enumerate(validation_dataloder):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            # Forward pass through the model
            x_pred = model(x)

            actual.append(y.cpu().numpy())
            predicted.append(x_pred.cpu().numpy())

    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    predicted = predicted > 0.90

    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

    cm_display.plot()
    plt.show()