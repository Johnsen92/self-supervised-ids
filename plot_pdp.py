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
from classes.statistics import PDPlot

def feature_string(features):
	feature_names_string = ''
	for _, ft in features.items():
		feature_names_string += '_' + ft
	return feature_names_string

parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', '--config_file', help='Config file for PD Plot', required=True)
parser.add_argument('-D', '--data_directory', help='Data directory for input pickle files', required=True)
parser.add_argument('-c', '--id_compare', default=None, help='Data directory for comparison input pickle files')
parser.add_argument('-i', '--id', help='ID for the run to be plotted', required=True)
parser.add_argument('-O', '--output_directory', default='./plots/pdp/', help='Output directory')
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

for features in config['features']:
	# Comprise feature string
	feature_names_string = feature_string(features)

	# Comprise file name
	file_name = args.data_directory + '/pdp/' + args.id + '/' + dataroot_filename + '_pdp' + feature_names_string + '.pickle'
	print(file_name)

	# Load pickle
	with open(file_name, "rb") as f:
		loaded = pickle.load(f)
	results_by_attack_number, feature_names, feature_values_by_attack_number = loaded["results_by_attack_number"], loaded["feature_names"], loaded["feature_values_by_attack_number"]
	print(int([k for k, _ in features.items()][0]))
	#print(feature_values_by_attack_number[int([k for k, _ in features.items()][0])])
	
	# Init PD plot
	pdp = PDPlot(
		results_by_attack_number = results_by_attack_number,
		feature_values_by_attack_number = feature_values_by_attack_number, 
		feature_names = features,
		mapping = mapping, 
		plot_dir = 'plots/pdp/' + args.id + '/', 
		output_basename = dataroot_filename
	)

	# If compare ID was provided, load comparison data
	if not args.id_compare is None:
		with open(file_name, "rb") as f:
			loaded = pickle.load(f)
		results_by_attack_number_compare, feature_names_compare, feature_values_by_attack_number_compare = loaded["results_by_attack_number"], loaded["feature_names"], loaded["feature_values_by_attack_number"]
		pdp_compare = PDPlot(
			results_by_attack_number = results_by_attack_number_compare,
			feature_values_by_attack_number = feature_values_by_attack_number_compare, 
			feature_names = features,
			mapping = mapping, 
			plot_dir = 'plots/pdp/' + args.id + '/', 
			output_basename = dataroot_filename
		)

	# Plot each attack type
	for attack_type in config['categories']:
		pdp.plot(attack_type)
