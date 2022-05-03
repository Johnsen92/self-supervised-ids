import sys
from torch.utils.data import random_split, DataLoader
from classes import utils, datasets, statistics
from classes.statistics import Stats, Epoch
from classes.datasets import Flows, FlowsSubset, Packets
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
from classes.utils import make_dir, rm_dir
import csv


def main(args):
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random_seed = args.random_seed

    # Set number of threads used by PyTorch
    torch.set_num_threads(args.n_threads)

    # Make output folders
    make_dir(args.trees_dir)
    make_dir(args.stats_dir)
    make_dir(args.plots_dir)
    make_dir(args.cache_dir)

    # Init hyperparameters
    data_filename = os.path.basename(args.data_file)[:-7]

    # Load feature description
    features_file = f'{os.path.split(args.data_file)[0]}/{data_filename}_features.json'
    assert os.path.isfile(features_file), f'{features_file} not found...'
    with open(features_file, 'r') as f:
        features = json.load(f)

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    
    tree_cache_file = 'tree_tuple'

    depth_start = args.max_depth if not args.depth_analysis or args.id_only else 1
    depth_analysis = []

    for depth in range(depth_start, args.max_depth+1):
        # ID and unique ID (with timestamp) for this run
        id = f'dt_{data_filename}_rn{random_seed}_tp{args.train_percent}_md{depth}_tc{args.target_category if args.target_category != -1 else "ALL"}'
        uid = f'{timestamp}_{id}'

        if args.id_only:
            return id
        # Init cache
        general_cache = utils.Cache(cache_dir=args.cache_dir, key_prefix=id, disabled=args.no_cache, label='LSTM Cache')
        if general_cache.exists(tree_cache_file):
            decision_tree, dtc, stats, run_stats, feature_names = general_cache.load(tree_cache_file)
        else:
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
            unused_size = n_samples - validation_size - training_size

            # correct for rounding error
            if (validation_size + training_size) - n_samples == 1:
                validation_size -= 1

            # Split dataset into pretraining, training and validation set
            if unused_size > 10:
                train_data, val_data, _ = subset.split([training_size, validation_size, unused_size], stratify=True)
            else:
                train_data, val_data = subset.split([training_size, validation_size], stratify=True)

            # Gather model parameters for statistic
            model_parameters = {
                'Max depth' : depth
            }

            # Initialize statistics
            run_stats = Stats(
                stats_dir = args.stats_dir,
                benign = args.benign_category,
                category_mapping = dataset.mapping,
                proxy_task = 'NONE',
                pretrain_percent = 0,
                train_percent = args.train_percent,
                val_percent = args.val_percent,
                n_epochs = 0,
                n_epochs_pretraining = 0,
                batch_size = 0,
                learning_rate = 0,
                model_parameters = model_parameters,
                random_seed = random_seed,
                subset = ''
            )
            
            stats = {}
            feature_names = [v for v in features.values()]

            # # Init model
            dtc = DecisionTreeClassifier(random_state=0, max_depth=depth)

            # Fit data
            print(f'Fitting decision tree with max. depth {depth}...', end='')
            start = timer()
            decision_tree = dtc.fit(train_data.x_catted, train_data.y_catted)
            end = timer()

            val_score = decision_tree.score(val_data.x_catted, val_data.y_catted)
            train_score = decision_tree.score(train_data.x_catted, train_data.y_catted)
            print(f'took {end - start} seconds with a validation accuracy score of {(val_score*100.0):.2f}%.')

            run_stats.new_epoch(
                epoch=0, 
                training_time=statistics.formatTime(int(end - start)), 
                training_loss=0
            )

            # Validation data loader
            raw_packet_dataset = Packets(val_data.x_catted, val_data.y_catted, val_data.c_catted)
            val_loader = DataLoader(dataset=raw_packet_dataset, batch_size=args.val_batch_size, shuffle=True, num_workers=args.n_worker_threads, drop_last=False)

            # Validate dtc
            print(f'Validating DTC with ID {id}...', end='')
            for data, targets, categories in val_loader:
                predicted = torch.from_numpy(dtc.predict(data))
                

                # Evaluate results
                run_stats.last_epoch.add_batch(predicted, targets.view(-1), categories.view(-1))
            print(f'done')

            # Calculate stats
            stats['max_depth'] = depth
            stats['depth'] = decision_tree.get_depth()
            stats['fitting_time s'] = round(end - start, 2)
            stats['val. accuracy %'] = round(val_score*100.0, 4)
            stats['train. accuracy %'] = round(train_score*100.0, 4)
            stats['packets_benign'] = FlowsSubset(val_data, dataset.mapping, ditch=[-1, args.benign_category]).n_packets
            stats['packets_attack'] = FlowsSubset(val_data, dataset.mapping, ditch=([-1, args.target_category] if args.target_category != -1 else [args.benign_category])).n_packets
            stats['packets_total'] = stats['packets_benign'] + stats['packets_attack']
            stats['benign_rate %'] = round(float(stats['packets_benign'])/float(stats['packets_total'])*100.0, 4)
            stats['attack_rate %'] = round(float(stats['packets_attack'])/float(stats['packets_total'])*100.0, 4)
            stats['above_guessing'] = stats['val. accuracy %'] > stats['benign_rate %']

            general_cache.save(tree_cache_file, (decision_tree, dtc, stats, run_stats, feature_names), msg='Storing tree tuple')

        depth_analysis.append((depth, id, stats))

    # Save depth analysis
    if args.depth_analysis:
        depth_analysis_filename = f'{args.stats_dir}/{id}_depth_analysis.csv'
        header = ['max. depth', 'accuracy', 'fitting t']
        with open(depth_analysis_filename, 'w+', newline='') as f_da:
            f_da_csv = csv.writer(f_da, delimiter=',', quotechar='|')
            f_da_csv.writerow(header)
            for depth, id, stats in depth_analysis:
                f_da_csv.writerow([depth, f'{stats["val. accuracy %"]}%', f'{stats["fitting_time s"]}s'])
    else:
        # Save DT
        r = export_text(decision_tree, feature_names=feature_names)
        if args.output_file == '':
            out_f = f'{args.trees_dir}/{id}.txt'
            out_f_plot = f'{args.plots_dir}/{id}.png'
        else:
            out_f = f'{args.trees_dir}/{args.output_file}'
            out_f_plot = f'{args.plots_dir}/{args.output_file[:-4]}' + '.png'

        # Write to output file
        with open(out_f, 'w+') as f:
            f.write(f'max. depth: {args.max_depth}, fitting time: {stats["fitting_time s"]:.2f}s, accuracy: {stats["val. accuracy %"]:.3f}%, {stats["benign_rate %"]:.3f}% benign samples, {stats["attack_rate %"]:.3f}% attack samples\n')
            f.write(r)

        # Save stats
        stats_file = f'{args.stats_dir}/{id}_stats.txt'
        with open(stats_file, 'w+') as f:
            f.write(str(stats))

        # # Save run stats
        run_stats_file = f'{id}_run_stats.csv'
        run_stats.save_stats(file_name=run_stats_file)

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


    

