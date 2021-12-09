import argparse
import sys
import csv
import os
import json
import pickle
import argparse
from classes.statistics import PDPlot, NeuronPlot
from classes.utils import TransformerArgumentParser, LSTMArgumentParser, ProxyTask
from main_lstm import main as train_lstm
from main_trans import main as train_trans
from tqdm import tqdm
import errno
import re
from enum import Enum
from datetime import datetime
import shutil
import copy
import traceback

class Mode(Enum):
    ALL = 0,
    STATS = 1,
    CLASS = 2,

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

def make_dir(dir):
    if not os.path.exists(dir):
        try:
                os.makedirs(dir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def rm_dir(dir):
    shutil.rmtree(dir, ignore_errors=True)

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

def generate_tables(entries, mode, data_dir, out_dir, order=None):
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

def get_group_description(group, group_description_file):
    with open(group_description_file, newline='') as f:
        group_descriptions = { f'{r[0]}_{r[1]}':(r[2], r[3]) for r in [row for row in csv.reader(f, delimiter=',', quotechar='"')][1:]}
    return group_descriptions[group]

def string_replace(file, string_replace_list):
    for v,k in string_replace_list.items():
        os.system(f"sed -i 's/\<{v}\>/{k}/g' {file}")

parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', '--parameter_file', help='File with table of parameters', required=True)
parser.add_argument('-m', '--model', help='Only return lines of the chosen model', required=True)
parser.add_argument('-S', '--stats_dir', help='Output directory', required=True)
parser.add_argument('-p', '--proxy_tasks', nargs='+', type=lambda proxy_task: ProxyTask[proxy_task], choices=list(ProxyTask), default=[], help='List of proxy tasks')
parser.add_argument('-N', '--neuron_data_dir', default='./data/neurons/', help='Folder for neuron activation plot configuration files')
parser.add_argument('-P', '--pdp_data_dir', default='./data/pdp/', help='Folder for PDP configuration files')
parser.add_argument('-d', '--debug', action='store_true', help='Debug flag')
parser.add_argument('-G', '--group_description', default='./groups_lstm.csv', help='CSV file containing caption and labels for groups occuring in run config')
parser.add_argument('-R', '--string_replace', default='./result_string_replace.csv', help='CSV file containing a list of strings to replace in output files')
args = parser.parse_args(sys.argv[1:])

CSV_EXPERIMENT_INDEX = 0
CSV_GROUP_INDEX = 1
CSV_MODEL_INDEX = 2
EXPECTED_RESULTS_FILES = 4

i = 0
table_groups = {}
while True:
    # Props if you can read this line... (just generates a dictionary out of unique entries separated by '|' in groups column)
    with open(args.parameter_file, newline='') as param_file_csv:
        groups = { k:[] for k in list(set([row[CSV_GROUP_INDEX].split('|')[i] for row in [row for row in csv.reader(param_file_csv, delimiter=',', quotechar='"')][2:] if len(row[CSV_GROUP_INDEX].split('|')) > i and row[CSV_GROUP_INDEX].split('|')[i] != ''])) }
        if len(groups.items()) == 0:
            break
        table_groups.update(groups)
        i += 1

with open(args.string_replace, newline='') as string_replace_file_csv:
    string_replace_list = { row[0]: row[1] for row in csv.reader(string_replace_file_csv, delimiter=',', quotechar='"') if len(row) == 2 }

pdp_groups = copy.deepcopy(table_groups)
neuron_ids = []
ids = []
param_file_csv = open(args.parameter_file, newline='')
rows = [row for row in csv.reader(param_file_csv, delimiter=',', quotechar='"')]
for i, row in enumerate(rows):
    if i == 1:
        parameters = row[3:]
    elif i == 0:
        labels = row
    elif row[CSV_MODEL_INDEX] == args.model:
        parameter_string, parameter_list = build_parameters(parameters, row[3:])
        if row[CSV_MODEL_INDEX] == 'lstm':
            train = train_lstm
            model_parser = LSTMArgumentParser(parameter_list)
        elif row[CSV_MODEL_INDEX] == 'transformer':
            model_parser = TransformerArgumentParser(parameter_list)
            train = train_trans
        dataroot_basename = os.path.basename(row[1])[:-7]
        base_dir = f'{args.stats_dir}/{row[CSV_MODEL_INDEX]}'
        stats_dir = f'{base_dir}/stats/'
        log_dir = f'{base_dir}/runs/'
        errorlog_dir = f'{args.stats_dir}/logs/'
        make_dir(errorlog_dir)
        add_parameter(parameter_list, '-S', stats_dir)
        add_parameter(parameter_list, '-L', log_dir)
        if args.debug:
            add_parameter(parameter_list, '--debug')
        model_args = model_parser.parse_args(parameter_list)
        if len(args.proxy_tasks) != 0 and not model_args.proxy_task in args.proxy_tasks:
            continue
        add_parameter(parameter_list, '--id_only')
        model_args_id_only = model_parser.parse_args(parameter_list)
        id = train(model_args_id_only)
        experiment = row[CSV_EXPERIMENT_INDEX]
        ids.append(id)
        if not model_args.neuron_config is None:
            neuron_ids.append((id, model_args.neuron_config))
        for group in row[CSV_GROUP_INDEX].split('|'): 
            if group != '':
                table_groups[group].append((id, experiment))
        if not model_args.pdp_config is None:
            for group in row[CSV_GROUP_INDEX].split('|'): 
                if group != '':
                    pdp_groups[group].append((id, model_args.pdp_config))
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

# Generate CSV Tables
table_dir = f'{base_dir}/tables/'
rm_dir(table_dir)
for k, group_entries in table_groups.items():
    if len(group_entries) <= 1:
        continue
    group_dir = f'{table_dir}/{k}'
    print(f'Generating tables for group {k}...', end='')
    generate_tables(group_entries, Mode.ALL, stats_dir, group_dir, order=ids)
    print('done.')

# Generate Latex Tables
path = os.walk(table_dir)
for root, dir, files in path:
    for file in files:
        csv_file = os.path.join(root, file)
        string_replace(csv_file, string_replace_list)
        if 'class_comparison' in file:
            group_prefix = 'class'
        elif 'stats_comparison' in file:
            group_prefix = 'stats'
        
        group_name = csv_file.split('/')[-2]
        group = f'{group_prefix}_{group_name}'

        group_label, group_caption = get_group_description(group, args.group_description)
        os.system(f'python3 ./script/tably.py {csv_file} -o {csv_file[:-4]}.tex -l "{group_label}" -c "{group_caption}"')

plots_dir = f'{base_dir}/plots/'
rm_dir(plots_dir)
make_dir(plots_dir)

# # Generate Neuron Activation Plots
# neuron_dir = f'{plots_dir}/neuron/'
# make_dir(neuron_dir)
# if len(neuron_ids) > 0:
#     for id, config in neuron_ids:
#         plot_neurons([id], args.neuron_data_dir, neuron_dir, config, 'pre')

# # Generate Partial Dependency Plots
# pdp_dir = f'{plots_dir}/pdp/'
# make_dir(pdp_dir)
# for group, pdp_ids in pdp_groups.items():
#     if len(pdp_ids) == 0:
#         continue
#     assert len(set([config for _, config in pdp_ids])) == 1, 'Different PDP configs used for PDP data...'
#     ids = [id for id, _ in pdp_ids]
#     config = [config for _, config in pdp_ids][0]
#     plot_pdp(ids, args.pdp_data_dir, pdp_dir, config)





