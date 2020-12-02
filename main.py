import argparse
import sys
import pickle
from torch.utils.data import random_split, DataLoader
from torch import optim, nn
from timeit import default_timer as timer
from datetime import timedelta
from classes import datasets, lstm, statistics, utils
import math
import torchvision
import torch
import os.path


# Define argument parser
parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', '--data_file', help='Pickle file containing the training data', required=True)
parser.add_argument('-g', '--gpu', action='store_true', help='Train on GPU if available')
parser.add_argument('-t', '--train', action='store_true', help='Force training even if cache file exists')
parser.add_argument('-d', '--debug', action='store_true', help='Debug flag')
parser.add_argument('-c', '--cache_dir', default='./cache/', help='Cache folder')
parser.add_argument('-s', '--stats_dir', default='./stats/', help='Statistics folder')
parser.add_argument('-e', '--n_epochs', default=10, help='Number of epochs')
parser.add_argument('-b', '--batch_size', default=32, help='Batch size')
parser.add_argument('-p', '--train_percent', default=90, help='Training percentage')
parser.add_argument('-l', '--hidden_size', default=512, help='Size of hidden layers')
parser.add_argument('-n', '--n_layers', default=3, help='Number of LSTM layers')
parser.add_argument('--no_cache', action='store_true', help='Flag to disable cache')

args = parser.parse_args(sys.argv[1:])

# Define hyperparameters
data_filename = os.path.basename(args.data_file)
learning_rate = 0.001
output_size = 2

# Define cache
key_prefix = data_filename[:-7] + f'_hs{args.hidden_size}_bs{args.batch_size}_ep{args.n_epochs}_tp{args.train_percent}'
cache = utils.Cache(cache_dir=args.cache_dir, md5=True, key_prefix=key_prefix, no_cache=args.no_cache)

# Load dataset and normalize data, or load from cache
if not cache.exists('dataset'):
    dataset = datasets.Flows(data_pickle=args.data_file, cache=cache)
    cache.save('dataset', dataset)
else:
    print('(Cache) Loading normalized dataset...')
    dataset = cache.load('dataset')

# Create data loaders
n_samples = len(dataset)
training_size = (n_samples*args.train_percent) // 100
validation_size = n_samples - training_size
train, val = random_split(dataset, [training_size, validation_size])
train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows, drop_last=True)
val_loader = DataLoader(dataset=val, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows, drop_last=True)

if torch.cuda.is_available() and args.gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Define model
data, labels, categories = dataset[0]
input_size = data.size()[1]
model = lstm.LSTM(input_size, args.hidden_size, output_size, args.n_layers, args.batch_size, device).to(device)

# Train model if no cache file exists or the train flag is set, otherwise load cached model
chache_file_name = args.cache_dir + key_prefix + '_trained_model.pickle'
if not os.path.isfile(chache_file_name) or args.train:
    # Define loss
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Define statistics object
    stats = statistics.Stats(
        stats_dir = args.stats_dir,
        n_samples = n_samples,
        train_percent = args.train_percent / 100.0,
        n_epochs = args.n_epochs,
        batch_size = args.batch_size,
        learning_rate = learning_rate
    )

    # Train the model
    print('Training model...')
    start = step = timer()
    n_total_steps = len(train_loader)
    losses = []
    monitoring_interval = n_samples * args.n_epochs // (1000 * args.batch_size) 
    for epoch in range(args.n_epochs):
        loss_sum = []
        for i, (data, labels, categories) in enumerate(train_loader): 

            # Move data to selected device 
            data = data.to(device)
            labels = labels.to(device)
            categories = categories.to(device)

            # Forward pass
            outputs = model(data)
            #print(outputs.size())
            #print(labels.size())
            #print(labels)
            loss = criterion(outputs, labels)
            loss_sum.append(loss.item())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate time left and save avg. loss of last interval
            if (i+1) % monitoring_interval == 0:
                avg_loss = sum(loss_sum)/len(loss_sum)
                last_step = step
                step = timer()
                n_batches = len(train_loader)
                interval_time = step - last_step
                sample_time = float(interval_time)/float(monitoring_interval)
                time_left = sample_time * float(n_batches * args.n_epochs - epoch * n_batches - i)/3600.0
                time_left_h = math.floor(time_left)
                time_left_m = math.floor((time_left - time_left_h)*60.0)
                print (f'Epoch [{epoch+1}/{args.n_epochs}], Step [{i+1}/{n_total_steps}], Avg. Loss: {avg_loss:.4f}, Time left: {time_left_h} h {time_left_m} m')
                losses.append(avg_loss)
                loss_sum = []

            # Break after x for debugging
            if args.debug and i == (1000 // args.batch_size):
                break
    print('...done')        

    # Get stats
    end = timer()
    stats.start_time = start
    stats.end_time = end
    stats.losses = losses


    # Store trained model
    print('Storing model to cache...',end='')
    torch.save(model.state_dict(), chache_file_name)
    print('done')

    # Store statistics object
    print('Storing statistics to cache...',end='')
    cache.save('stats', stats)
    print('done')
else:
    # Load cached model
    print('(Cache) Loading trained model...',end='')
    model.load_state_dict(torch.load(chache_file_name))
    model.eval()
    print('done')

    # Load statistics object
    print('(Cache) Loading statistics object...',end='')
    stats = cache.load('stats')
    print('done')

# Validate model
print('Validating model...')
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_false_positive = 0
    n_false_negative = 0
    for i, (data, labels, categories) in enumerate(train_loader):

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
        n_false_negative += (predicted < labels).sum().item()
        n_false_positive += (predicted > labels).sum().item()
        assert n_correct == n_samples - n_false_negative - n_false_positive

        # Break after x for debugging
        if args.d and i == 1000:
            break

    # Calculate statistics
    acc = 100.0 * n_correct / n_samples
    false_p = 100.0 * n_false_positive/(n_samples - n_correct)
    false_n = 100.0 * n_false_negative/(n_samples - n_correct)

    # Save and cache statistics
    stats.n_false_negative = n_false_negative
    stats.n_false_positive = n_false_positive
    print(f'Accuracy with validation size {((1.0-args.train_percent)*100):.2f}% of data samples: {stats.getAccuracy()*100}%, False p.: {false_p}%, False n.: {false_n}%')
    if not args.debug:
        stats.saveStats()
        stats.saveLosses()
        stats.plotLosses()
    

