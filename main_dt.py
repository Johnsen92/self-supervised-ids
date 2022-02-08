import sys
from torch.utils.data import random_split, DataLoader
from classes import utils
from classes.datasets import Flows, FlowsSubset
import torch
import os.path
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text, plot_tree
from timeit import default_timer as timer
import json

def main(args):
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random_seed = args.random_seed

    # Set number of threads used by PyTorch
    torch.set_num_threads(args.n_threads)

    # Init hyperparameters
    data_filename = os.path.basename(args.data_file)[:-7]

    # Load feature description
    features_file = f'{os.path.split(args.data_file)[0]}/{data_filename}_features.json'
    assert os.path.isfile(features_file), f'{features_file} not found...'
    with open(features_file, 'r') as f:
        features = json.load(f)

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ID and unique ID (with timestamp) for this run
    id = f'dt_{data_filename}_rn{random_seed}_md{args.max_depth}_tc{args.target_category if args.target_category != -1 else "ALL"}'
    uid = f'{timestamp}_{id}'

    if args.id_only:
        return id

    # Init cache
    general_cache = utils.Cache(cache_dir=args.cache_dir, key_prefix=id, disabled=args.no_cache, label='LSTM Cache')

    # Extended stats directory for this run
    extended_stats_dir = (args.stats_dir if args.stats_dir[-1] == '/' else args.stats_dir + '/') + id + '/'

    # Load dataset and normalize data, or load from cache
    cache_filename = f'dataset_normalized_{data_filename}'
    if general_cache.exists(cache_filename, no_prefix=True) and not args.no_cache:
        dataset = general_cache.load(cache_filename, no_prefix=True, msg='Loading decision tree dataset')
    else:
        dataset = Flows(data_pickle=args.data_file, cache=general_cache, max_length=args.max_sequence_length)
    
    if args.target_category == -1:
        subset = FlowsSubset(dataset, dataset.mapping)
    else:
        subset = FlowsSubset(dataset, dataset.mapping, ditch=[-1, args.benign_category, args.target_category])

    # Number of samples
    n_samples = len(subset)

    # Split dataset into training and validation parts
    validation_size = int(round((n_samples * args.val_percent) / 1000.0))
    training_size = int(round((n_samples * args.train_percent) / 1000.0))

    # correct for rounding error
    if (validation_size + training_size) - n_samples == 1:
        validation_size -= 1

    # Split dataset into pretraining, training and validation set
    train_data, val_data = subset.split([training_size, validation_size], stratify=True)

    stats = {}
    feature_names = [v for v in features.values()]

    # Init model
    dtc = DecisionTreeClassifier(random_state=0, max_depth=args.max_depth)

    # Fit data
    print(f'Fitting decision tree with max. depth {args.max_depth}...', end='')
    start = timer()
    decision_tree = dtc.fit(train_data.x_catted, train_data.y_catted)
    end = timer()
    val_score = decision_tree.score(val_data.x_catted, val_data.y_catted)
    train_score = decision_tree.score(train_data.x_catted, train_data.y_catted)
    print(f'took {end - start} seconds with a validation accuracy score of {(val_score*100.0):.2f}%.')
    r = export_text(decision_tree, feature_names=feature_names)

    if args.output_file == '':
        out_f = f'{args.stats_dir}/{uid}.txt'
        out_f_plot = f'{args.stats_dir}/{uid}.png'
    else:
        out_f = f'{args.stats_dir}/{args.output_file}'
        out_f_plot = out_f[:-4] + '.png'

    benign_samples = subset.subset_count[args.benign_category]
    attack_samples = subset.subset_count[args.target_category]
    total_samples = benign_samples + attack_samples

    # Calculate stats
    stats['max_depth'] = args.max_depth
    stats['depth'] = decision_tree.get_depth()
    stats['fitting_time s'] = round(end - start, 2)
    stats['val. accuracy %'] = round(val_score*100.0, 4)
    stats['train. accuracy %'] = round(train_score*100.0, 4)
    stats['packets_benign'] = FlowsSubset(val_data, dataset.mapping, ditch=[-1, args.benign_category]).n_packets
    stats['packets_attack'] = FlowsSubset(val_data, dataset.mapping, ditch=[-1, args.target_category]).n_packets
    stats['packets_total'] = stats['packets_benign'] + stats['packets_attack']
    stats['benign_rate %'] = round(float(stats['packets_benign'])/float(stats['packets_total'])*100.0, 4)
    stats['attack_rate %'] = round(float(stats['packets_attack'])/float(stats['packets_total'])*100.0, 4)
    stats['above_guessing'] = stats['val. accuracy %'] > stats['benign_rate %']

    # Write to output file
    with open(out_f, 'w+') as f:
        f.write(f'max. depth: {args.max_depth}, fitting time: {(end - start):.2f}s, accuracy: {stats["val. accuracy %"]:.3f}%, {stats["benign_rate %"]:.3f}% benign samples, {stats["attack_rate %"]:.3f}% attack samples\n')
        f.write(r)

    # If plot flag is set, plot decision tree
    if args.plot:
        fig = plt.figure(figsize=(args.x, args.y))
        _ = plot_tree(dtc, 
                        feature_names=feature_names,  
                        class_names=['benign', 'attack'],
                        filled=True)
        fig.savefig(out_f_plot)

    # Remove temp directories
    general_cache.clean()
    print(f'Run with ID \"{id}\" has ended successfully')
    return stats

if __name__=="__main__":
    # Init argument parser
    parser = utils.DTArgumentParser(description='Decision Tree based IDS')
    args = parser.parse_args(sys.argv[1:])
    main(args)


    

