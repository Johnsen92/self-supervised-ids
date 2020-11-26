import argparse
import sys
import pickle
from classes import datasets
from torch.utils.data import random_split, DataLoader
from torch import optim, nn
from timeit import default_timer as timer
from datetime import timedelta
from classes import lstm, statistics, utils
import math
import torchvision
import torch
import os.path

# Define argument parser
parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', help='Pickle file containing the training data')
parser.add_argument('-g', action='store_true', help='Train on GPU if available')
parser.add_argument('-t', action='store_true', help='Force training even if cache file exists')
parser.add_argument('-d', action='store_true', help='Debug flag')
parser.add_argument('-c', default="./cache/", help='Cache folder')
parser.add_argument('-s', default="./stats/", help='Statistics folder')
args = parser.parse_args(sys.argv[1:])

# Define hyperparameters
learning_rate = 0.0001
batch_size = 16
n_epochs = 1
training_percentage = 0.9

# Load dataset and create data loaders
dataset = datasets.Flows(args.f)
n_samples = len(dataset)
training_size = math.floor(n_samples*training_percentage)
validation_size = n_samples - training_size
train, val = random_split(dataset, [training_size, validation_size])
#train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=12)
train_loader = DataLoader(dataset=train, sampler=datasets.FlowBatchSampler(dataset=train, batch_size=batch_size, drop_last=True), num_workers=0)
#val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True, num_workers=12)
val_loader = DataLoader(dataset=train, sampler=datasets.FlowBatchSampler(dataset=val, batch_size=batch_size, drop_last=True), num_workers=0)

sampler = datasets.FlowBatchSampler(dataset=train, batch_size=batch_size, drop_last=True)
for (data, labels, categories) in sampler:
    print(data.size())
    print(labels.size())
    print(categories.size())

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

    # Define statistics object
    stats = statistics.Stats(
        n_samples = n_samples,
        training_percentage = training_percentage,
        n_epochs = n_epochs,
        batch_size = batch_size,
        learning_rate = learning_rate
    )

    interval = math.floor(n_samples * n_epochs / 1000)

    # Train the model
    print("Training model...")
    start = step = timer()
    n_total_steps = len(train_loader)
    losses = []
    for epoch in range(n_epochs):
        loss_sum = []
        for i, (data, labels, categories) in enumerate(train_loader): 

            # Pad data if batch size is greater one
            data_padded, data_lengths = torch.nn.utils.rnn.pad_sequence(data)

            # Move data to selected device 
            data_padded = data_padded.to(device)
            labels = labels.to(device)
            categories = categories.to(device)

            # Forward pass
            outputs = model(data_padded)
            loss = criterion(outputs, labels)
            loss_sum.append(loss.item())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % interval == 0:
                avg_loss = sum(loss_sum)/len(loss_sum)
                last_step = step
                step = timer()
                interval_time = step - last_step
                sample_time = float(interval_time)/float(interval)
                time_left = sample_time * float(n_samples * n_epochs - epoch * n_samples - i)/3600
                time_left_h = math.floor(time_left)
                time_left_m = math.floor((time_left - time_left_h)*60)
                print (f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{n_total_steps}], Avg. Loss: {avg_loss:.4f}, Time left: {time_left_h} h {time_left_m} m')
                losses.append(avg_loss)
                loss_sum = []

            # Break after x for debugging
            if args.d and i == 10000:
                break

    # Get stats
    end = timer()
    stats.start_time = start
    stats.end_time = end
    stats.losses = losses


    # Store trained model
    print("Storing model to cache...",end='')
    torch.save(model.state_dict(), chache_file_name)
    print("done")

    # Store statistics object
    print("Storing statistics to cache...",end='')
    with open(args.f[:-7]+"_stats.pickle", "wb") as f:
        f.write(pickle.dumps(stats))
    print("done")
else:
    # Load cached model
    print("Loading cached model...",end='')
    model.load_state_dict(torch.load(chache_file_name))
    model.eval()
    print("done")

    # Load statistics object
    print("Loading statistics object...",end='')
    with open(args.f[:-7]+"_stats.pickle", "rb") as f:
        stats = pickle.load(f)
    print("done")

# Validate model
print("Validating model...")
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_false_positive = 0
    n_false_negative = 0
    for i, (data, labels, categories) in enumerate(train_loader):
        
        # Pad data if batch size is greater one
        data_padded, data_lengths = torch.nn.utils.rnn.pad_sequence(data)

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
        if predicted.item() == 0.0 and labels.item() == 1.0:
            n_false_negative += 1
        elif predicted.item() == 1.0 and labels.item() == 0.0:
            n_false_positive += 1

        # Break after x for debugging
        if args.d and i == 1000:
            break

    acc = 100.0 * n_correct / n_samples
    false_p = 100.0 * n_false_positive/(n_samples - n_correct)
    false_n = 100.0 * n_false_negative/(n_samples - n_correct)
    print(f"Accuracy with validation size {math.ceil(1-training_percentage)*100}% of data samples: {acc}%, False p.: {false_p}%, False n.: {false_n}%")
    stats.n_false_negative = n_false_negative
    stats.n_false_positive = n_false_positive
    stats.saveStats()
    stats.saveLosses()
    stats.plotLosses()

