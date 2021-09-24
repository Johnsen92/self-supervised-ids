#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import json
import pickle
from collections import Counter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import argparse
from classes.statistics import PDPlot, NeuronPlot

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
		if neuron_data.label == 'NONE':
			continue
		neuron_data_list.append(neuron_data)

		# Load postfix pickle
		file_name = args.data_directory + id + '_' + postfix + '.pickle'
		with open(file_name, 'rb') as f:
			neuron_data = pickle.load(f)
		neuron_data_list.append(neuron_data)

		neuron_plot = NeuronPlot(config, mapping, neuron_data_list, plot_dir=out_dir, use_titles=True)
		neuron_plot.plot_all()


parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', '--config_file', help='Config file for PD Plot', required=True)
parser.add_argument('-D', '--data_directory', help='Data directory for input pickle files', required=True)
parser.add_argument('-i', '--ids', nargs='+', help='IDs for the run to be plotted', required=True)
parser.add_argument('-O', '--output_directory', default='./plots/pdp/', help='Output directory')
parser.add_argument('-p', '--postfix', default=None, help='If set, produces tables to compare <ID> to <ID>_<postfix>')
args = parser.parse_args(sys.argv[1:])

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



