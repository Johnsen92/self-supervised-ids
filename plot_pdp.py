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
parser.add_argument('-f', '--config_file', help='Config file for PD Plot', required=True)
parser.add_argument('-D', '--data_directory', help='Data directory for input pickle files', required=True)
parser.add_argument('-i', '--ids', nargs='+', help='IDs for the run to be plotted', required=True)
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

pd_data_list = []

for id in args.ids:
	# Comprise file name
	file_name = args.data_directory + id + '.pickle'
	print(file_name)

	# Load pickle
	with open(file_name, "rb") as f:
		pd_data = pickle.load(f)
	pd_data_list.append(pd_data)

pdp = PDPlot(config, mapping, pd_data_list)
pdp.plot_all()

