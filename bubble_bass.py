import argparse
import sys
import csv
import os
import json
import pickle
import re
from classes import statistics

# Some nasty metacode
def update_epoch(old_epoch):
    new_epoch = statistics.Epoch(old_epoch.epoch, old_epoch.class_stats, old_epoch.training_time)
    new_epoch.n_true_positive = old_epoch.n_true_positive
    new_epoch.n_samples_counted = old_epoch.n_samples_counted
    new_epoch.n_true_positive = old_epoch.n_true_positive
    new_epoch.n_true_negative = old_epoch.n_true_negative
    new_epoch.n_false_positive = old_epoch.n_false_positive
    new_epoch.n_false_negative = old_epoch.n_false_negative
    new_epoch.training_loss = old_epoch.training_loss
    new_epoch.validation_loss = old_epoch.validation_loss
    return new_epoch

def update_stats(old_stats):
    new_stats = statistics.Stats(
        train_percent = old_stats.train_percent, 
        val_percent = old_stats.val_percent, 
        n_epochs = old_stats.n_epochs,
        batch_size = old_stats.batch_size, 
        learning_rate = old_stats.learning_rate, 
        category_mapping = old_stats.category_mapping,
        benign = old_stats.benign,
        model_parameters = old_stats.model_parameters,
        n_epochs_pretraining =  old_stats.n_epochs_pretraining, 
        pretrain_percent = old_stats.pretrain_percent, 
        proxy_task = old_stats.proxy_task, 
        title = old_stats.title, 
        random_seed = old_stats.random_seed, 
        subset = old_stats.subset,
        stats_dir = old_stats.stats_dir
    )
    new_stats.stats_dir = old_stats.stats_dir
    new_stats.train_percent = old_stats.train_percent
    new_stats.pretrain_percent = old_stats.pretrain_percent
    new_stats.proxy_task = old_stats.proxy_task
    new_stats.val_percent = old_stats.val_percent
    new_stats.n_epochs = old_stats.n_epochs
    new_stats.n_epochs_pretraining = old_stats.n_epochs_pretraining
    new_stats.batch_size = old_stats.batch_size
    new_stats.learning_rate = old_stats.learning_rate  
    new_stats.losses = old_stats.losses
    new_stats.mapping = old_stats.mapping
    new_stats.benign = old_stats.benign
    new_stats.model_parameters = old_stats.model_parameters
    new_stats.random_seed = old_stats.random_seed
    new_stats.epochs = [update_epoch(epoch) for epoch in old_stats.epochs]
    new_stats.subset = old_stats.subset
    new_stats.title = old_stats.title
    return new_stats

parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-I', '--input_directory', help='Directory to skim for pickles', required=True)
parser.add_argument('-O', '--output_directory', help='Output directory. Default is the same as input directory')
parser.add_argument('-r', '--regex', help='Regex to search for in directory. If not provided, all pickles are updated')

def main(args):
    path = os.walk(args.input_directory)
    for root, dir, files in path:
        for file in files:
            if not re.search(args.regex, file) is None:
                with open(os.path.join(root, file), 'rb') as f:
                    pkl = pickle.load(f)
                if isinstance(pkl, statistics.Epoch):
                    pkl = update_epoch(pkl)
                #with open (f'{args.output_directory}/{file_name}', 'wb+') as f:
                #    f.write(pickle.dumps(pkl))

if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    main(args)