import argparse
import sys
import csv
import os
import json
import pickle
import argparse
from classes.utils import Cache
from classes.datasets import Flows, FlowsSubset
from classes.statistics import PDPlot, NeuronPlot
from classes.utils import TransformerArgumentParser, LSTMArgumentParser, ProxyTask, DTArgumentParser, rm_dir, make_dir
from main_lstm import main as train_lstm
from main_trans import main as train_trans
from main_dt import main as train_dt
import re
from enum import Enum
from datetime import datetime
import copy
import traceback
import matplotlib.pyplot as plt
import numpy as np

_DT_LABELS = {
    "depth" : "dth",
    "fitting_time s" : "fit.t.[s]",
    "val. accuracy %" : "val.acc.",
    "train. accuracy %" : "tr.acc.",
    "benign_rate %" : "benign[%]",
    "attack_rate %" : "attack[%]",
    "above_guessing" : "> guess"
}

class Mode(Enum):
    ALL = 0,
    STATS = 1,
    CLASS = 2,
    TRAINING = 3,
    VALIDATION = 4

    def __str__(self):
        return self.name

def build_parameters(parameters, values):
    assert len(parameters) == len(values)
    par_str = ''
    par_list = []
    for i in range(len(parameters)):
        if values[i] != '':
            par_str += f'{parameters[i]} {values[i]} '
            par_list.append(parameters[i])
            par_list.append(values[i])

    return par_str, par_list

def add_parameter(parameter_list, parameter, value = None):
    parameter_list.append(parameter)
    if value != None:
        parameter_list.append(value)
    return parameter_list

def compare_ids(ids, config, mapping, out_dir, base_name):
	neuron_data_list = []
	for id in ids:
		# Comprise file name
		file_name = args.data_directory + id + '.pickle'
		print(file_name)

		# Load pickle
		with open(file_name, 'rb') as f:
			neuron_data = pickle.load(f)
		neuron_data_list.append(neuron_data)

	neuron_plot = NeuronPlot(config, mapping, neuron_data_list, plot_dir=out_dir, base_name=base_name)
	neuron_plot.plot_all()

def compare_postfix(ids, postfix, config, mapping, out_dir, input_dir, base_name):
	for id in ids:
		neuron_data_list = []
		# Comprise file name

		# Load pickle
		file_name = input_dir + id + '.pickle'
		with open(file_name, 'rb') as f:
			neuron_data = pickle.load(f)
		neuron_data_list.append(neuron_data)
		if neuron_data.label.startswith('NONE'):
			continue

		# Load postfix pickle
		file_name = input_dir + id + '_' + postfix + '.pickle'
		with open(file_name, 'rb') as f:
			neuron_data = pickle.load(f)
		neuron_data_list.append(neuron_data)

		neuron_plot = NeuronPlot(config, mapping, neuron_data_list, plot_dir=out_dir, use_titles=True, compare=True, base_name=base_name)
		neuron_plot.plot_all()

def plot_neurons(ids, input_dir, output_dir, config_file, postfix):
    dataroot_basename = config_file.split('_')[0]
    dataroot_filename = os.path.basename(dataroot_basename)

    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Load category mapping
    with open(dataroot_basename + "_categories_mapping.json", "r") as f:
        categories_mapping_content = json.load(f)
    categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
    reverse_mapping = {v: k for k, v in mapping.items()}

    if postfix is None:
        compare_ids(ids, config, mapping, output_dir, dataroot_filename)
    else:
        compare_postfix(ids, postfix, config, mapping, output_dir, input_dir, dataroot_filename)

def plot_pdp(ids, input_dir, output_dir, config_file):
    dataroot_basename = config_file.split('_')[0]
    dataroot_filename = os.path.basename(dataroot_basename)

    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Load category mapping
    with open(dataroot_basename + "_categories_mapping.json", "r") as f:
        categories_mapping_content = json.load(f)
    categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
    reverse_mapping = {v: k for k, v in mapping.items()}

    pd_data_list = []

    for id in ids:
        # Comprise file name
        file_name = input_dir + id + '.pickle'
        print(file_name)

        # Load pickle
        with open(file_name, "rb") as f:
            pd_data = pickle.load(f)
        pd_data_list.append(pd_data)

    pdp = PDPlot(config, mapping, pd_data_list, plot_dir=output_dir, base_name=dataroot_filename)
    pdp.plot_all()

def remake_dir(dir):
    rm_dir(dir)
    make_dir(dir)

def get_label_from_path(path):
    return re.search(r'\_xy(\w+)', path).group(1)

def stats_table(input_files, output_file, headers):
    out_labels = ['']
    out_rows = []
    first = True
    for file in input_files:
        with open(file, 'r') as f:
            table = csv.reader(f, delimiter=',')
            for i, row in enumerate(table):
                if len(row) == 2 and row[0]=='Proxy task':
                    #out_labels.append(row[1])
                    out_labels.append(headers[file])
                if first:
                    out_rows.append(row)
                elif len(row) >= 2:
                    out_rows[i].append(row[1])
            first = False

    with open(output_file, 'w+') as out_f:
        writer = csv.writer(out_f, delimiter=',')
        writer.writerow(out_labels)
        for row in out_rows:
            writer.writerow(row)

def dt_stats_table(input_files, output_file, headers):
    out_labels = ['']
    out_rows = []
    json_map = {
        'depth': 'Actual depth',
        'fitting_time s' : 'Fittings time [s]'
    }
    include = ['Accuracy', 'Detection rate', 'Precision', 'Specificity', 'F1-Measure', 'False alarm rate', 'Missed alarm rate']

    first = True
    for file in input_files:
        stats_file = file[:-14] + '_stats.txt'
        stats_file_sanizized = stats_file[:-4] + '.json'
        with open(stats_file_sanizized, 'w+') as jfs:
            with open(stats_file, 'r') as sf:
                jfs.write(sf.read().replace('\'', '"').replace('\n', '').replace('True','\"True\"').replace('False','\"False\"'))
        with open(stats_file_sanizized) as jf:
            stats = json.load(jf)
        i = 0
        for key, val in json_map.items():
            out_rows.append([])
            if first:
                out_rows[i].append(val)
            out_rows[i].append(stats[key])
            i += 1
        out_rows.append([])
        i += 1
        with open(file, 'r') as f:
            table = csv.reader(f, delimiter=',')
            for row in table:
                if len(row) > 0 and row[0] == 'Training percentage':
                    out_labels.append(row[1])
                if len(row) > 0 and row[0] in include:
                    out_rows.append([])
                    if first:
                        out_rows[i].append(row[0])
                    out_rows[i].append(row[1])
                    i += 1
        first = False

    print(out_rows)
    with open(output_file, 'w+') as out_f:
        writer = csv.writer(out_f, delimiter=',')
        writer.writerow(out_labels)
        for row in out_rows[:-20]:
            writer.writerow(row)

def dt_class_table(input_files, output_file, headers):
    pass
    # out_labels = ['']
    # out_rows = []
    # first = True
    # for file in input_files:
    #     if first:
    #         out_labels.append('')
    #     out_labels.append(headers[file])
    #     with open(file, 'r') as f:
    #         table = csv.reader(f, delimiter=',')
    #         for i, row in enumerate(table):
    #             if first:
    #                 if len(row) == 5:
    #                     out_rows.append(row[:2])
    #                     out_rows[i].append(row[-1])
    #                 else:
    #                     out_rows.append(row)
    #             elif len(row) == 5:
    #                 out_rows[i].append(row[-1])
                
    #         first = False

    # with open(output_file, 'w+') as out_f:
    #     writer = csv.writer(out_f, delimiter=',')
    #     writer.writerow(out_labels)
    #     for row in out_rows:
    #         writer.writerow(row)

def class_table(input_files, output_file, headers):
    out_labels = ['']
    out_rows = []
    first = True
    for file in input_files:
        if first:
            out_labels.append('')
        out_labels.append(headers[file])
        with open(file, 'r') as f:
            table = csv.reader(f, delimiter=',')
            for i, row in enumerate(table):
                if first:
                    if len(row) == 5:
                        out_rows.append(row[:2])
                        out_rows[i].append(row[-1])
                    else:
                        out_rows.append(row)
                elif len(row) == 5:
                    out_rows[i].append(row[-1])
                
            first = False

    with open(output_file, 'w+') as out_f:
        writer = csv.writer(out_f, delimiter=',')
        writer.writerow(out_labels)
        for row in out_rows:
            writer.writerow(row)

def sort_files(file_list, order=['NONE', 'PREDICT', 'OBSCURE', 'MASK', 'AUTO', 'ID', 'COMPOSITE']):
    files_sorted = []
    for item in order:
        filtered = [file for file in file_list if not re.search(re.escape(item), file) is None]
        if len(filtered) > 0:  
            for f in filtered:
                files_sorted.append(f)
    return files_sorted

def plot_graphs(input_files, output_file, headers):
    # plot lines
    plt.figure(figsize=(12, 6.75))
    for file in input_files:
        proxy_task = get_label_from_path(file)
        with open(file, 'r') as f:
            x, y = zip(*[(int(row[0])+1, float(row[1])) for row in csv.reader(f, delimiter=',') if len(row) == 2])
            plt.plot(list(x), list(y), label = f'{headers[file]} ({proxy_task})')
    plt.legend()
    #plt.ylim(0, max(y_second))
    plt.savefig(output_file, bbox_inches = 'tight', pad_inches = 0.1)
    plt.clf()

def get_group_description(group, group_description_file):
    with open(group_description_file, newline='') as f:
        group_descriptions = { f'{r[0]}_{r[1]}':(r[2], r[3]) for r in [row for row in csv.reader(f, delimiter=',', quotechar='"')][1:]}
    return group_descriptions[group]

def string_replace(file, string_replace_list):
    for v,k in string_replace_list.items():
        os.system(f"sed -i 's/\<{v}\>/{k}/g' {file}")

def gen_dt_table(entries, mode, data_dir, out_dir, model, order=None):
    regex = {}
    if mode == Mode.STATS or mode == Mode.ALL: 
        regex[Mode.STATS] = r"run\_stats.csv"
    if mode == Mode.CLASS or mode == Mode.ALL:
        regex[Mode.CLASS] = r"run\_stats\_class.csv"

    data = {}
    experiments = {}
    for k, rx in regex.items():
        data[k] = []
        path = os.walk(data_dir)
        for root, dir, files in path:
            for file in files:
                if model == 'dt':
                    match = re.match('(\w+_tc[a-zA-Z0-9]+)_\w+\.csv', file)
                    if not match:
                        continue
                    else:
                        id = match[1]
                else:
                    id = root.split('/')[-1]
                if id in [id for id, _ in entries] and not re.search(rx, file) is None:
                    path = os.path.join(root, file)
                    experiments[path] = [exp for i, exp in entries if i == id][0]
                    data[k].append(path)

    os.makedirs(out_dir, exist_ok=True)
    out_files = []

    for table, file_list in data.items():
        files_sorted = sort_files(file_list, order)
        if table == Mode.STATS:
            out_file = f'{out_dir}/stats_comparison_{str(mode)}.csv'
            dt_stats_table(files_sorted, out_file, experiments)
        elif table == Mode.CLASS:
            out_file = f'{out_dir}/class_comparison_{str(mode)}.csv'
            dt_class_table(files_sorted, out_file, experiments)
        out_files.append(out_file)

    out_files_str = ''
    for o_f in out_files:
        out_files_str += o_f + ' '

def gen_table(entries, mode, data_dir, out_dir, model, order=None):
    regex = {}
    if mode == Mode.STATS or mode == Mode.ALL: 
        regex[Mode.STATS] = r"^stats_"
    if mode == Mode.CLASS or mode == Mode.ALL:
        regex[Mode.CLASS] = r"class\_stats"

    data = {}
    experiments = {}
    for k, rx in regex.items():
        data[k] = []
        path = os.walk(data_dir)
        for root, dir, files in path:
            for file in files:
                id = root.split('/')[-1]
                if id in [id for id, _ in entries] and not re.search(rx, file) is None:
                    path = os.path.join(root, file)
                    experiments[path] = [exp for i, exp in entries if i == id][0]
                    data[k].append(path)

    os.makedirs(out_dir, exist_ok=True)
    out_files = []

    for table, file_list in data.items():
        files_sorted = sort_files(file_list, order)
        if table == Mode.STATS:
            out_file = f'{out_dir}/stats_comparison_{str(mode)}.csv'
            stats_table(files_sorted, out_file, experiments)
        elif table == Mode.CLASS:
            out_file = f'{out_dir}/class_comparison_{str(mode)}.csv'
            class_table(files_sorted, out_file, experiments)
        out_files.append(out_file)

    out_files_str = ''
    for o_f in out_files:
        out_files_str += o_f + ' '

def gen_graphs(entries, mode, data_dir, out_dir, order=None):
    regex = {}
    if mode == Mode.TRAINING or mode == Mode.ALL: 
        regex[Mode.TRAINING] = r"^training\_losses"
    if mode == Mode.VALIDATION or mode == Mode.ALL:
        regex[Mode.VALIDATION] = r"validation\_losses"

    data = {}
    experiments = {}
    for k, rx in regex.items():
        data[k] = []
        path = os.walk(data_dir)
        for root, dir, files in path:
            for file in files:
                id = root.split('/')[-1]
                if id in [id for id, _ in entries] and not re.search(rx, file) is None:
                    path = os.path.join(root, file)
                    experiments[path] = [exp for i, exp in entries if i == id][0]
                    data[k].append(path)

    os.makedirs(out_dir, exist_ok=True)
    out_files = []

    for graph, file_list in data.items():
        files_sorted = sort_files(file_list, order)
        if graph == Mode.TRAINING:
            out_file = f'{out_dir}/training_loss.png' 
        elif graph == Mode.VALIDATION:
            out_file = f'{out_dir}/validation_loss.png'
        plot_graphs(files_sorted, out_file, experiments)
        out_files.append(out_file)

    out_files_str = ''
    for o_f in out_files:
        out_files_str += o_f + ' '

def gen_tables(table_dir, stats_dir, model, table_groups, ids, group_description, replace):
    # Generate CSV Tables
    rm_dir(table_dir)
    for k, group_entries in table_groups.items():
        if len(group_entries) <= 1:
            continue
        group_dir = f'{table_dir}/{k}'
        print(f'Generating tables for model {model} group {k}...', end='')
        if model == 'dt':
            gen_dt_table(group_entries, Mode.ALL, stats_dir, group_dir, model, order=ids)
        else:
            gen_table(group_entries, Mode.ALL, stats_dir, group_dir, model, order=ids)
        print('done.')

    # Calculate scale factors for latex tables
    group_scale_factor = { k: 12.0/max(float(len(v))+8.0,12.0) for k,v in table_groups.items() }

    # Generate Latex Tables
    path = os.walk(table_dir)
    for root, dir, files in path:
        for file in files:
            csv_file = os.path.join(root, file)
            string_replace(csv_file, replace)
            if 'class_comparison' in file:
                group_prefix = 'class'
            elif 'stats_comparison' in file:
                group_prefix = 'stats'
            
            group_name = csv_file.split('/')[-2]
            group = f'{group_prefix}_{group_name}'

            group_label, group_caption = get_group_description(group, group_description)
            os.system(f'python3 ./script/tably.py {csv_file} -o {csv_file[:-4]}.tex -x {group_scale_factor[group_name]} -l "{group_label}" -c "{group_caption}"')
    
def gen_dataset_stats_analysis(datasets):
    # Generate data analysis tables of datasets
    headers = ['Feature', '#', 'mean ds', 'mean cat', 'std ds', 'std cat', 'z', 'p']
    data_analysis_dir = f'{args.stats_dir}/dataset/'
    make_dir(data_analysis_dir)
    datasets = list(set(datasets))
    cache = Cache(cache_dir=args.cache_dir, label='Results Cache')
    for ds, benign_category in datasets:
        dataset_name = os.path.basename(ds)[:-7]
        dataset_mapping_name = f'{dataset_name}_mapping'
        features_file = f'{os.path.split(ds)[0]}/{dataset_name}_features.json'
        assert os.path.isfile(features_file), f'{features_file} not found...'
        with open(features_file, 'r') as f:
            features = json.load(f)

        data_analysis_dataset_dir = f'{data_analysis_dir}/{dataset_name}'
        make_dir(data_analysis_dataset_dir)

        # Load datatset
        cache_filename = f'dataset_normalized_{dataset_name}'
        if cache.exists(cache_filename, no_prefix=True):
            dataset = cache.load(cache_filename, no_prefix=True, msg='Loading normalized dataset')
        else:
            dataset = Flows(data_pickle=args.data_file, cache=cache, max_length=args.max_sequence_length)
        dataset = FlowsSubset(dataset, dataset.mapping)

        if not cache.exists(dataset_mapping_name):
            cache.save(dataset_mapping_name, dataset.mapping, no_prefix=True)

        # Calculate statistical data
        dataset_variance = dataset.subset_variance
        dataset_std = dataset.subset_std
        dataset_means = dataset.subset_means
        for cat_label, cat_num in dataset.mapping.items():
            # Calculate statistical data
            statistical_analysis_file = f'{data_analysis_dataset_dir}/st_{cat_num}_{cat_label}.csv'
            if not os.path.isfile(statistical_analysis_file):
                subset = FlowsSubset(dataset, dataset.mapping, ditch=[-1, cat_num])
                z, p = subset.z_test(dataset)
                table_values = np.round(np.stack((dataset_means, subset.subset_means, dataset_std, subset.subset_std, z, p), axis=-1), 2)
                with open(statistical_analysis_file, 'w+') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(headers)
                    for i, row in enumerate(table_values):
                        writer.writerow([features[str(i)]] + [str(i)] + row.tolist())
            else:
                print(f'Skipping statistical analysis for dataset {dataset_name}, category {cat_label}...')

def gen_dataset_dtc_depth_analysis(datasets):
    MAX_DEPTH = 25
    for ds, benign_category in datasets:
        dataset_name = os.path.basename(ds)[:-7]
        data_analysis_dir = f'{args.stats_dir}/dataset/'
        make_dir(data_analysis_dir)
        data_analysis_dataset_dir = f'{data_analysis_dir}/{dataset_name}'
        make_dir(data_analysis_dataset_dir)

        # Generate decision trees
        parameter_list = []
        add_parameter(parameter_list, '-m', str(MAX_DEPTH))
        add_parameter(parameter_list, '-f', ds)
        add_parameter(parameter_list, '-S', data_analysis_dataset_dir)
        add_parameter(parameter_list, '--random_seed', str(500))
        add_parameter(parameter_list, '-c', str(benign_category))
        add_parameter(parameter_list, '--depth_analysis')
        dt_arg_parser = DTArgumentParser('Decision Tree Argument Parser').parse_args(parameter_list)
        train_dt(dt_arg_parser)
        add_parameter(parameter_list, '--id_only')
        dt_arg_parser = DTArgumentParser('Decision Tree Argument Parser').parse_args(parameter_list)
        id = train_dt(dt_arg_parser)
        dt_summary_file = f'{data_analysis_dataset_dir}/{id}_depth_analysis.csv'
        dt_summary_tex_file = f'{dt_summary_file[:-4]}.tex'
        if os.path.exists(dt_summary_tex_file):
            os.remove(dt_summary_tex_file)
        os.system(f'python3 ./script/tably.py {dt_summary_file} -o {dt_summary_tex_file} -l "table:results:dtc:depth_analysis_{dataset_name}" -c "Performance analysis of the DTC for benign classification of dataset {dataset_name}"')

def gen_dataset_dtc_analysis(datasets):
    # Generate data analysis tables of datasets
    headers = ['Feature', '#', 'mean ds', 'mean cat', 'std ds', 'std cat', 'z', 'p']
    data_analysis_dir = f'{args.stats_dir}/dataset/'
    make_dir(data_analysis_dir)
    datasets = list(set(datasets))
    cache = Cache(cache_dir=args.cache_dir, label='Results Cache')
    for ds, benign_category in datasets:
        dataset_name = os.path.basename(ds)[:-7]
        dataset_mapping_name = f'{dataset_name}_mapping'
        features_file = f'{os.path.split(ds)[0]}/{dataset_name}_features.json'
        assert os.path.isfile(features_file), f'{features_file} not found...'
        with open(features_file, 'r') as f:
            features = json.load(f)

        data_analysis_dataset_dir = f'{data_analysis_dir}/{dataset_name}'
        make_dir(data_analysis_dataset_dir)

        # Calculate decision tree
        if cache.exists(dataset_mapping_name, no_prefix=True):
            dataset_mapping = cache.load(dataset_mapping_name, no_prefix=True)
        else:
            raise Exception(f'Dataset mapping \'{dataset_mapping}\' must exist in cache')
        dtc_dataset_dir = f'{data_analysis_dir}/{dataset_name}'
        make_dir(dtc_dataset_dir)
        dtc_stats_dir = f'{dtc_dataset_dir}/stats'
        dtc_plots_dir = f'{dtc_dataset_dir}/plots'
        dtc_tables_dir = f'{dtc_dataset_dir}/tables'
        dtc_trees_dir = f'{dtc_dataset_dir}/trees'
        dt_stats_summary = {}
        MAX_DEPTH = 5
        dt_summary_name = f'dt_{dataset_name}_md{MAX_DEPTH}_summary'
        dt_summary_file = f'{data_analysis_dir}/{dt_summary_name}.csv'
        for i, (cat_label, cat_num) in enumerate(dataset_mapping.items()):
            dt_file_name = (f'dt_{MAX_DEPTH}_{cat_num}_{cat_label}.txt')
            if cat_num != int(benign_category):
                # Generate decision trees
                parameter_list = []
                add_parameter(parameter_list, '-m', str(MAX_DEPTH))
                add_parameter(parameter_list, '-f', ds)
                add_parameter(parameter_list, '-t', str(cat_num))
                add_parameter(parameter_list, '-S', dtc_stats_dir)
                add_parameter(parameter_list, '-T', dtc_trees_dir)
                add_parameter(parameter_list, '-P', dtc_plots_dir)
                add_parameter(parameter_list, '-o', dt_file_name.replace(' ','_'))
                add_parameter(parameter_list, '--random_seed', str(500))
                add_parameter(parameter_list, '-c', str(benign_category))
                add_parameter(parameter_list, '--plot')
                dt_arg_parser = DTArgumentParser('Decision Tree Argument Parser').parse_args(parameter_list)
                dt_stats_summary[(cat_label, cat_num)] = train_dt(dt_arg_parser)
            print(f'----------------------------------------------------------------{i+1}/{len(dataset_mapping.items())}---------------------------------------------------------------------------')

        with open(dt_summary_file, 'w+') as f:
            writer = csv.writer(f, delimiter=',')
            first = True
            for (cat_label, cat_num), stats in dt_stats_summary.items():
                if first:
                    writer.writerow(['Category', '#'] + [_DT_LABELS[k] for k, _ in stats.items() if k in _DT_LABELS.keys()])
                    first = False
                writer.writerow([cat_label, cat_num] + [v for k, v in stats.items() if k in _DT_LABELS.keys()])

        dt_summary_tex_file = f'{dt_summary_file[:-4]}.tex'
        if os.path.exists(dt_summary_tex_file):
            os.remove(dt_summary_tex_file)
        os.system(f'python3 ./script/tably.py {dt_summary_file} -o {dt_summary_tex_file} -l "table:results:dtc:{dataset_name}" -c "Results of the DTC discerning between benign packets and packets of a certain attack type of dataset {dataset_name}"')

def gen_loss_plots(plots_dir, table_groups):
    # Generate loss progression graphs
    graphs_dir = f'{plots_dir}/losses/'
    make_dir(graphs_dir)
    for k, group_entries in table_groups.items():
        if len(group_entries) <= 1:
            continue
        group_dir = f'{graphs_dir}/{k}'
        print(f'Generating graphs for group {k}...', end='')
        gen_graphs(group_entries, Mode.ALL, stats_dir, group_dir, order=ids)
        print('done.')
    pass

def gen_neuron_plots(plots_dir, neuron_ids, neuron_data_dir):
    # Generate Neuron Activation Plots
    neuron_dir = f'{plots_dir}/neuron/'
    remake_dir(neuron_dir)
    if len(neuron_ids) > 0:
        for id, config in neuron_ids:
            plot_neurons([id], neuron_data_dir, neuron_dir, config, 'pre')

def gen_pdp_plots(plots_dir, pdp_groups, pdp_data_dir):
    # Generate Partial Dependency Plots
    pdp_dir = f'{plots_dir}/pdp/'
    make_dir(pdp_dir)
    for group, pdp_ids in pdp_groups.items():
        if len(pdp_ids) == 0:
            continue
        assert len(set([config for _, config in pdp_ids])) == 1, 'Different PDP configs used for PDP data...'
        ids = [id for id, _ in pdp_ids]
        config = [config for _, config in pdp_ids][0]
        plot_pdp(ids, pdp_data_dir, pdp_dir, config)


parser = argparse.ArgumentParser(description='Results script for self-seupervised machine learning IDS')
parser.add_argument('-t', '--transformer_experiment_file', help='File with table of transformer experiments', required=True)
parser.add_argument('-l', '--lstm_experiment_file', help='File with table of lstm experiments', required=True)
parser.add_argument('-d', '--dtc_experiment_file', help='File with table of dtc experiments')
parser.add_argument('-S', '--stats_dir', help='Output directory', required=True)
parser.add_argument('-p', '--proxy_tasks', nargs='+', type=lambda proxy_task: ProxyTask[proxy_task], choices=list(ProxyTask), default=[], help='List of proxy tasks')
parser.add_argument('-N', '--neuron_data_dir', default='./data/neurons/', help='Folder for neuron activation plot configuration files')
parser.add_argument('-P', '--pdp_data_dir', default='./data/pdp/', help='Folder for PDP configuration files')
parser.add_argument('--debug', action='store_true', help='Debug flag')
parser.add_argument('-R', '--string_replace', default='./result_string_replace.csv', help='CSV file containing a list of strings to replace in output files')
parser.add_argument('-C', '--cache_dir', default='./cache/', help='Cache folder')
parser.add_argument('--no_cache', action='store_true', help='Cache folder')
args = parser.parse_args(sys.argv[1:])

CSV_EXPERIMENT_INDEX = 0
CSV_GROUP_INDEX = 1
EXPECTED_RESULTS_FILES = 4

experiments = [('dt', args.dtc_experiment_file), ('lstm', args.lstm_experiment_file), ('transformer', args.transformer_experiment_file)]
#experiments = [('dt', args.dtc_experiment_file)]

table_groups = {}
pdp_groups = {}
neuron_ids = {}
ids = {}

with open(args.string_replace, newline='') as string_replace_file_csv:
    string_replace_list = { row[0]: row[1] for row in csv.reader(string_replace_file_csv, delimiter=',', quotechar='"') if len(row) == 2 }


for model, parameter_file in experiments:
    base_dir = f'{args.stats_dir}/{model}'
    stats_dir = f'{base_dir}/stats/'
    log_dir = f'{base_dir}/runs/'
    plots_dir = f'{base_dir}/plots/'
    errorlog_dir = f'{args.stats_dir}/logs/'
    trees_dir = f'{base_dir}/trees/'
    table_dir = f'{base_dir}/tables/'
    table_groups[model] = {}
    i = 0
    while True:
        # Props if you can read this line... (just generates a dictionary out of unique entries separated by '|' in groups column)
        with open(parameter_file, newline='') as param_file_csv:
            groups = { k:[] for k in list(set([row[CSV_GROUP_INDEX].split('|')[i] for row in [row for row in csv.reader(param_file_csv, delimiter=',', quotechar='"')][2:] if len(row[CSV_GROUP_INDEX].split('|')) > i and row[CSV_GROUP_INDEX].split('|')[i] != ''])) }
            if len(groups.items()) == 0:
                break
            table_groups[model].update(groups)
            i += 1
    
    pdp_groups[model] = copy.deepcopy(table_groups[model])
    neuron_ids[model] = []
    ids[model] = []
    param_file_csv = open(parameter_file, newline='')
    rows = [row for row in csv.reader(param_file_csv, delimiter=',', quotechar='"')]
    datasets = []
    make_dir(errorlog_dir)
    for i, row in enumerate(rows):
        if i == 1:
            parameters = row[3:]
        elif i == 0:
            labels = row
        else:
            parameter_string, parameter_list = build_parameters(parameters, row[3:])
            if model == 'lstm':
                train = train_lstm
                model_parser = LSTMArgumentParser(parameter_list)
            elif model == 'transformer':
                model_parser = TransformerArgumentParser(parameter_list)
                train = train_trans
            elif model == 'dt':
                model_parser = DTArgumentParser(parameter_list)
                train = train_dt
            add_parameter(parameter_list, '-S', stats_dir)
            if model != 'dt':
                add_parameter(parameter_list, '-L', log_dir)
            else:
                add_parameter(parameter_list, '-T', trees_dir)
                add_parameter(parameter_list, '-P', plots_dir)
            if args.debug:
                add_parameter(parameter_list, '--debug')
            model_args = model_parser.parse_args(parameter_list)
            if model != 'dt' and (len(args.proxy_tasks) != 0 and not model_args.proxy_task in args.proxy_tasks):
                continue
            add_parameter(parameter_list, '--id_only')
            model_args_id_only = model_parser.parse_args(parameter_list)
            id = train(model_args_id_only)
            experiment = row[CSV_EXPERIMENT_INDEX]
            ids[model].append(id)
            datasets.append((model_args.data_file, model_args.benign_category))
            if model != 'dt' and not model_args.neuron_config is None:
                neuron_ids[model].append((id, model_args.neuron_config))
            for group in row[CSV_GROUP_INDEX].split('|'): 
                if group != '':
                    table_groups[model][group].append((id, experiment))
            if model != 'dt' and not model_args.pdp_config is None:
                for group in row[CSV_GROUP_INDEX].split('|'): 
                    if group != '':
                        pdp_groups[model][group].append((id, model_args.pdp_config))
            stats_dir_extended = f'{stats_dir}{id}/'
            print(f'----------------------------------------------------------------{i-1}/{len(rows)-2}---------------------------------------------------------------------------')
            try:
                if not os.path.exists(stats_dir_extended):
                    print(f'Make results for ID {id}...')
                    train(model_args)
                elif len(os.listdir(stats_dir_extended)) != EXPECTED_RESULTS_FILES:
                    print(f'Make results for ID {id}...')
                    os.system(f'rm -r "{stats_dir_extended}" -f')
                    train(model_args)
                else:
                    print(f'Skipping {id}...')
            except not KeyboardInterrupt:
                with open(f'{errorlog_dir}/{datetime.now():%Y%m%d}_{id}.txt','w+') as f:
                    f.write(traceback.format_exc())
                    f.write(f'Error in run {id}')
                print(f'An error occured during run {id}')

    gen_tables(table_dir, stats_dir, model, table_groups[model], ids[model], f'groups_{model}.csv', string_replace_list)
    if model != 'dt':
        gen_loss_plots(plots_dir, table_groups[model])
        #gen_pdp_plots(args.plots_dir, pdp_groups, args.pdp_data_dir)
        # if model == 'lstm':
        #     gen_neuron_plots(plots_dir, neuron_ids[model], args.neuron_data_dir)
        

gen_dataset_dtc_analysis(datasets)

datasets = list(set(datasets))

gen_dataset_dtc_depth_analysis(datasets)

#gen_dataset_stats_analysis(datasets)







