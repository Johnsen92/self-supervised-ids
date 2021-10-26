import argparse
import sys
import csv

def build_parameter_string(parameters, values):
    par_str = ''
    for i in range(1, len(parameters), 1):
        if values[i] != '/':
            par_str += f'{parameters[i]} {values[i]} '

    return par_str

parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', '--parameter_file', help='File with table of parameters', required=True)
parser.add_argument('-m', '--model', help='Only return lines of the chosen model', required=True)
parser.add_argument('-i', '--index', type=int, default=0, help='Index of row which is to be returned')
parser.add_argument('-c', '--count', action='store_true', help='Count how many parameter rows exist for given model')
args = parser.parse_args(sys.argv[1:])

# Using readlines()
param_file_csv = open(args.parameter_file, newline='')
rows = csv.reader(param_file_csv, delimiter=',', quotechar='"')

# List of options
parameters = []

# List of labels
labels = []

count = 0
parameter_values = []

# Strips the newline character
for i, row in enumerate(rows):
    if i == 0:
        parameters = row
    elif i == 1:
        labels = row
    
    if row[0] == args.model:
        if args.index == count:
            parameter_values = row
        count += 1

assert args.index < count, f'Only {count} parameter rows for this model: index {args.index} out of range'

if args.count:
    print(count)
else:
    print(build_parameter_string(parameters, parameter_values))