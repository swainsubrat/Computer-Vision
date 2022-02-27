"""
Pytorch implementation of the Autoencoder
"""
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import numpy as np

from torch import nn
from utils import accuracy, line_plot
from dataloader import load_mnist
	
def save_checkpoint(epoch, classifier, optimizer, path='./models/checkpoint_mnist_reg.pth.tar'):
    state = {'epoch': epoch,
             'classifier': classifier,
             'optimizer': optimizer}

    filename = path
    torch.save(state, filename)


class ANN(nn.Module):
    """
    Encoder to encode the image into a hidden state
    """
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(784, 500),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(500, 10)
        )
    
    def forward(self, images):
        out = self.classifier(images)

        return out

if __name__ == "__main__":
    batch_size = 64
    epochs     = 10
    patience   = 6
    min_valid_loss = np.inf
    
    print(f"Using {device} as the accelerator")
    train_dataloader, valid_dataloader, test_dataloader = load_mnist(batch_size=batch_size, root='./data/')
    try:
        # try loading checkpoint
        checkpoint = torch.load('./models/checkpoint_mnist_reg.pth.tar')
        print("Found Checkpoint :)")
        classifier = checkpoint["classifier"]
        classifier.to(device)

    except:
        # train the model from scratch
        print("Couldn't find checkpoint :(")

        classifier = ANN()
        classifier.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-5)

        train_los = []
        valid_los = []
        train_acc  = []
        valid_acc  = []
        epoch_since_last_improve = 0
        for epoch in range(epochs):
            train_loss = 0.0
            y_hat, y = [], []
            classifier.train()
            for i, (image, label) in enumerate(train_dataloader):
                image.to(device)
                out = classifier(image)
                
                loss = criterion(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                if i % 100 == 0 and i != 0:
                    print(f"Epoch: [{epoch+1}][{i}/{len(train_dataloader)}] Loss: {loss.item(): .4f}")
                
                prob, idxs = torch.max(out, dim=1)
                y.extend(label.tolist())
                y_hat.extend(idxs.tolist())
            
            train_acc.append(accuracy(y, y_hat))

            valid_loss = 0.0
            classifier.eval()
            y_hat, y = [], []

            for i, (image, label) in enumerate(valid_dataloader):
                image.to(device)
                out = classifier(image)
                loss = criterion(out, label)
                valid_loss = loss.item() * batch_size

                prob, idxs = torch.max(out, dim=1)
                y.extend(label.tolist())
                y_hat.extend(idxs.tolist())
            
            valid_acc.append(accuracy(y, y_hat))
                
            print(
                f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader): .4f} \
                \t\t Validation Loss: {valid_loss / len(valid_dataloader): .4f}'
            )

            epoch_since_last_improve += 1
            if valid_loss < min_valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.4f}--->{valid_loss:.4f}) \t Saving The Model')
                min_valid_loss = valid_loss
                epoch_since_last_improve = 0
                save_checkpoint(epoch, classifier, optimizer)
            
            train_los.append(train_loss / len(train_dataloader))
            valid_los.append(valid_loss / len(valid_dataloader))
            # print(train_los)
            # print(train_acc)
            
            print(f"Epochs since last improvement: {epoch_since_last_improve}")
            if epoch_since_last_improve > patience:
                print("Early stopping! Breaking out of loop....")
                break
    
        line_plot(train_acc, valid_acc, "Accuracy vs Epochs", "Epochs", "Accuracy", "./plots/mnist_reg_acc.png")
        line_plot(train_los, valid_los, "Loss vs Epochs", "Epochs", "Loss", "./plots/mnist_reg_loss.png")

    y_hat, y = [], []
    for i, (image, label) in enumerate(test_dataloader):
        image.to(device)
        out = classifier(image)
        prob, idxs = torch.max(out, dim=1)

        y.extend(label.tolist())
        y_hat.extend(idxs.tolist())

    print(accuracy(y, y_hat))
