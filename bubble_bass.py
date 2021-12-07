import argparse
import sys
import csv
import os
import json
import pickle
import re
from classes import statistics

# Some nasty metacode
def update(pkl):
    if isinstance(pkl, statistics.Epoch):
        new_pkl = statistics.Epoch(pkl.epoch, pkl.class_stats, pkl.training_time)
        new_pkl.n_true_positive = pkl.n_true_positive
        new_pkl.n_samples_counted = pkl.n_samples_counted
        new_pkl.n_true_positive = pkl.n_true_positive
        new_pkl.n_true_negative = pkl.n_true_negative
        new_pkl.n_false_positive = pkl.n_false_positive
        new_pkl.n_false_negative = pkl.n_false_negative
        new_pkl.training_loss = pkl.training_loss
        new_pkl.validation_loss = pkl.validation_loss
    elif isinstance(pkl, statistics.Stats):
        pass
    return new_pkl

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
                pkl = update(pkl)
                #with open (f'{args.output_directory}/{file_name}', 'wb+') as f:
                #    f.write(pickle.dumps(pkl))

if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    main(args)