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

parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-c', '--config_file', help='Config file for PD Plot', required=True)
parser.add_argument('-D', '--data_directory', help='Data directory for input pickle files', required=True)
parser.add_argument('-i', '--id', help='ID for the run to be plotted', required=True)
parser.add_argument('-O', '--output_directory', default='./plots/pdp/', help='Output directory')
args = parser.parse_args(sys.argv[1:])

dataroot_basename = args.config_file.split('_')[0]
dataroot_filename = os.path.basename(dataroot_basename)

with open(args.config_file, 'r') as f:
    config = json.load(f)

with open(dataroot_basename + "_categories_mapping.json", "r") as f:
	categories_mapping_content = json.load(f)
categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
reverse_mapping = {v: k for k, v in mapping.items()}
print("reverse_mapping", reverse_mapping)

for features in config:
	feature_names_string = ''
	for _, ft in features.items():
		feature_names_string += '_' + ft
	file_name = args.data_directory + dataroot_filename + '_pdp_' + args.id + feature_names_string + '.pickle'
	print(file_name)
	with open(file_name, "rb") as f:
		loaded = pickle.load(f)
	results_by_attack_number, feature_names, feature_values_by_attack_number = loaded["results_by_attack_number"], loaded["feature_names"], loaded["feature_values_by_attack_number"]
	pdp = PDPlot(
		results_by_attack_number = results_by_attack_number,
		feature_values_by_attack_number = feature_values_by_attack_number, 
		feature_names = features,
		mapping = mapping, 
		plot_dir = 'plots/pdp/' + args.id + '/', 
		output_basename = dataroot_filename
	)
	for attack_type in [v for k, v in mapping.items()]:
		pdp.plot(attack_type)
