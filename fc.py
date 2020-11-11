import argparse
import sys
from classes import datasets
from torch.utils.data import random_split, DataLoader
from torch import optim, nn
from timeit import default_timer as timer
from datetime import timedelta
from classes import linear
import math
import torchvision
import torch

# define argument parser
parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', help='CSV File containing the training data')
parser.add_argument('-t', action='store_true', help='Flag that enables training the network')
args = parser.parse_args(sys.argv[1:])

# parse arguments
csv = args.f

# load dataset and dataloaders
dataset = datasets.CAIA(csv)
training_size = math.floor(dataset.__len__()*0.9)
validation_size = dataset.__len__() - training_size
train, val = random_split(dataset, [training_size, validation_size])
train_loader = DataLoader(dataset=train, batch_size=100, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val, batch_size=100, shuffle=True, num_workers=0)

# hyper params
nb_epochs = 5
lr = 0.01

# define model
nb_features = dataset[0][0].size(0)
model = linear.LinResNet(nb_features).cuda()

# define stochastic gradient descent optimizer
opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0)

# Loss function
loss = nn.CrossEntropyLoss()

# START time mesurement
start = timer()

# Training loop
print('Training...')
for epoch in range(nb_epochs):

    losses = list()
    accuracies = list()
    for batch in train_loader:
        x, y = batch

        # Step 1: Forward execution, execute logits
        l = model(x)

        # Step 2: Compute loss
        J = loss(l, y)

        # Step 3: Clean greadients
        model.zero_grad()

        # Step 4: Compute partial derivatives dJ/dW
        J.backward()

        # Step 5: Train
        opt.step()

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch + 1}, training loss: {torch.tensor(losses).mean():.2f}, testing accuracy: {torch.tensor(accuracies).mean():.2f}')

    # Validation loop
    losses = list()
    accuracies = list()
    for batch in val_loader:
        x, y = batch

        # Step 1: Forward execution, execute logits
        l = model(x)

        # Step 2: Compute loss
        J = loss(l, y)

        # Step 3: Clean greadients
        model.zero_grad()

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch + 1}, validation loss: {torch.tensor(losses).mean():.2f}, validation accuracy: {torch.tensor(accuracies).mean():.2f}')

    end = timer()

print('done')
print(f'Elapsed time: {timedelta(seconds=end-start)}')

