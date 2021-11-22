import argparse
import sys
import csv
import os
import subprocess
import json
import pickle
import argparse
from classes.statistics import PDPlot, NeuronPlot
from classes.utils import TransformerArgumentParser, LSTMArgumentParser, ProxyTask
from main_lstm import main as train_lstm
from main_trans import main as train_trans
from tqdm import tqdm
import time

def build_parameters(parameters, values):
    assert len(parameters) == len(values)
    par_str = ''
    par_list = []
    for i in range(2, len(parameters), 1):
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

def compare_ids(ids, config, mapping, out_dir):
	neuron_data_list = []
	for id in ids:
		# Comprise file name
		file_name = args.data_directory + id + '.pickle'
		print(file_name)

		# Load pickle
		with open(file_name, 'rb') as f:
			neuron_data = pickle.load(f)
		neuron_data_list.append(neuron_data)

	neuron_plot = NeuronPlot(config, mapping, neuron_data_list, plot_dir=out_dir)
	neuron_plot.plot_all()

def compare_postfix(ids, postfix, config, mapping, out_dir):
	for id in ids:
		neuron_data_list = []
		# Comprise file name

		# Load pickle
		file_name = args.data_directory + id + '.pickle'
		with open(file_name, 'rb') as f:
			neuron_data = pickle.load(f)
		neuron_data_list.append(neuron_data)
		if neuron_data.label.startswith('NONE'):
			continue

		# Load postfix pickle
		file_name = args.data_directory + id + '_' + postfix + '.pickle'
		with open(file_name, 'rb') as f:
			neuron_data = pickle.load(f)
		neuron_data_list.append(neuron_data)

		neuron_plot = NeuronPlot(config, mapping, neuron_data_list, plot_dir=out_dir, use_titles=True, compare=True)
		neuron_plot.plot_all()

def plot_neurons():
    dataroot_basename = args.config_file.split('_')[0]
    dataroot_filename = os.path.basename(dataroot_basename)

    # Load config
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # Load category mapping
    with open(dataroot_basename + "_categories_mapping.json", "r") as f:
        categories_mapping_content = json.load(f)
    categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
    reverse_mapping = {v: k for k, v in mapping.items()}

    if args.postfix is None:
        compare_ids(args.ids, config, mapping, args.output_directory)
    else:
        compare_postfix(args.ids, args.postfix, config, mapping, args.output_directory)

def plot_pdp(ids, config_file, data_directory, output_directory):
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
        file_name = data_directory + id + '.pickle'
        print(file_name)

        # Load pickle
        with open(file_name, "rb") as f:
            pd_data = pickle.load(f)
        pd_data_list.append(pd_data)

    pdp = PDPlot(config, mapping, pd_data_list, plot_dir=output_directory)
    pdp.plot_all()

parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', '--parameter_file', help='File with table of parameters', required=True)
parser.add_argument('-m', '--model', help='Only return lines of the chosen model', required=True)
parser.add_argument('-S', '--stats_dir', help='Output directory', required=True)
parser.add_argument('-p', '--proxy_tasks', nargs='+', type=lambda proxy_task: ProxyTask[proxy_task], choices=list(ProxyTask), default=[], help='List of proxy tasks')
parser.add_argument('-N', '--neuron_config_dir', help='Folder for neuron activation plot configuration files')
parser.add_argument('-P', '--pdp_config_dir', help='Folder for PDP configuration files')
parser.add_argument('-d', '--debug', action='store_true', help='Debug flag')
args = parser.parse_args(sys.argv[1:])

CSV_MODEL_INDEX = 1
CSV_GROUP_INDEX = 0
EXPECTED_RESULTS_FILES = 6

with open(args.parameter_file, newline='') as param_file_csv:
    # Props if you can read this line... (just generates a dictionary out of unique entries in groups column)
    groups = { k:[] for k in list(set([row[0] for row in [row for row in csv.reader(param_file_csv, delimiter=',', quotechar='"')][2:]])) }

ids = []
param_file_csv = open(args.parameter_file, newline='')
rows = [row for row in csv.reader(param_file_csv, delimiter=',', quotechar='"')]
for i, row in enumerate(rows):
    if i == 1:
        parameters = row
    elif i == 0:
        labels = row
    elif row[CSV_MODEL_INDEX] == args.model:
        parameter_string, parameter_list = build_parameters(parameters, row)
        if row[CSV_MODEL_INDEX] == 'lstm':
            train = train_lstm
            model_parser = LSTMArgumentParser(parameter_list)
        elif row[CSV_MODEL_INDEX] == 'transformer':
            model_parser = TransformerArgumentParser(parameter_list)
            train = train_trans
        dataroot_basename = os.path.basename(row[1])[:-7]
        stats_dir = f'{args.stats_dir}/{row[CSV_MODEL_INDEX]}/stats/'
        log_dir = f'{args.stats_dir}/{row[CSV_MODEL_INDEX]}/runs/'
        add_parameter(parameter_list, '-S', stats_dir)
        add_parameter(parameter_list, '-L', log_dir)
        add_parameter(parameter_list, '--debug')
        model_args = model_parser.parse_args(parameter_list)
        if len(args.proxy_tasks) != 0 and not model_args.proxy_task in args.proxy_tasks:
            continue
        add_parameter(parameter_list, '--id_only')
        model_args_id_only = model_parser.parse_args(parameter_list)
        id = train(model_args_id_only)
        ids.append(id)
        groups[row[CSV_GROUP_INDEX]].append(id)
        stats_dir_extended = f'{stats_dir}{id}/'
        print(f'----------------------------------------------------------------{i-1}/{len(rows-2)}---------------------------------------------------------------------------')
        if not os.path.exists(stats_dir_extended):
            print(f'Make results for ID {id}...')
            train(model_args)
        elif len(os.listdir(stats_dir_extended)) != EXPECTED_RESULTS_FILES:
            print(f'Make results for ID {id}...')
            os.system(f'rm -r {stats_dir_extended} -f')
            train(model_args)
        else:
            print(f'Skipping {id}...')

        

        
    