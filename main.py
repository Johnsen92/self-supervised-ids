import argparse
import sys
from classes import datasets
from torch.utils.data import random_split, DataLoader
from torch import optim, nn
from timeit import default_timer as timer
from datetime import timedelta
from classes import lstm
import math
import torchvision
import torch
import os.path

# Define argument parser
parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', help='Pickle file containing the training data')
parser.add_argument('-g', action='store_true', help='Train on GPU if available')
parser.add_argument('-t', action='store_true', help='Force training even if cache file exists')
args = parser.parse_args(sys.argv[1:])

# Define hyperparameters
learning_rate = 0.001
batch_size = 1
num_epochs = 1

# Load dataset and create data loaders
dataset = datasets.Flows(args.f)
training_size = math.floor(len(dataset)*0.9)
validation_size = len(dataset) - training_size
train, val = random_split(dataset, [training_size, validation_size])
train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True, num_workers=4)

#data, labels, categories = dataset[0]
#print(labels)
#print(data.size())
#print(labels.size())
#print(categories.size())
#print(categories)
#print(dataset.getCategories())

if torch.cuda.is_available() and args.g:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Define model
data, labels, categories = dataset[0]
input_size = data.size()[1]
hidden_size = 64
output_size = 2
num_layers = 1
model = lstm.LSTM(input_size, hidden_size, output_size, num_layers, batch_size, device).to(device)

# Train model if no cache file exists or the train flag is set, otherwise load cached model
chache_file_name = args.f[:-7]+"_trained_model.pickle"
if not os.path.isfile(chache_file_name) or args.t:
    # Define loss
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Train the model
    print("Training model...")
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (data, labels, categories) in enumerate(train_loader):  

            # Move data to selected device 
            data = data.to(device)
            labels = labels.to(device)
            categories = categories.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    # Store trained model
    print("Storing model to cache...",end='')
    torch.save(model.state_dict(), chache_file_name)
    print("done")
else:
    # Load cached model
    print("Loading cached model...",end='')
    model.load_state_dict(torch.load(chache_file_name))
    model.eval()
    print("done")

# Validate model
print("Validating model...")
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for (data, labels, categories) in val_loader:
        
        # Move data to selected device 
        data = data.to(device)
        labels = labels.to(device)
        categories = categories.to(device)

        # Forward pass
        outputs = model(data)

        # Max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy with validation size {validation_size*100}% of data samples: {acc}%')