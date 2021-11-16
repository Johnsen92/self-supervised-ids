import argparse
import sys
import csv
import os
import subprocess

def build_parameter_string(parameters, values):
    par_str = ''
    for i in range(1, len(parameters), 1):
        if values[i] != '/':
            par_str += f'{parameters[i]} {values[i]} '

    return par_str

parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-f', '--parameter_file', help='File with table of parameters', required=True)
parser.add_argument('-m', '--model', help='Only return lines of the chosen model', required=True)
parser.add_argument('-S', '--stats_dir', help='Output directory', required=True)
parser.add_argument('-p', '--proxy_tasks', nargs='+', help='List of proxy tasks', required=True)
args = parser.parse_args(sys.argv[1:])

stats_dir = (args.stats_dir if args.stats_dir[-1] == '/' else args.stats_dir + '/') + 'stats/'
param_file_csv = open(args.parameter_file, newline='')
rows = csv.reader(param_file_csv, delimiter=',', quotechar='"')
parameters = ''
for i, row in enumerate(rows):
    if i == 0:
        parameters = row
    elif i == 1:
        labels = row
    elif row[0] == args.model:
        parameter_string = build_parameter_string(parameters, row)
        if row[0] == 'LSTM':
            python_main = 'main_lstm.py'
        elif row[0] == 'Transformer':
            python_main = 'main_trans.py'
        for task in args.proxy_tasks:
            pt_parameter = f'-y {task} -s 800 -E 10' if task != 'NONE' else ''
            getId = subprocess.Popen(f'python3 {python_main} {parameter_string} {pt_parameter} --id_only', shell=True, stdout=subprocess.PIPE).stdout
            id = getId.read().decode('utf-8')[:-1]
            stats_dir_extended = f'{stats_dir}{id}/'
            if not os.path.exists(stats_dir_extended):
                print(f'Make results for ID {id}...')
                os.system(f'python3 main_lstm.py {parameter_string} {pt_parameter} -S {stats_dir}')
            elif len(os.listdir(stats_dir_extended)) != 4:
                print(f'Make results for ID {id}...')
                os.system(f'rm -r {stats_dir_extended} -f')
                os.system(f'python3 main_lstm.py {parameter_string} {pt_parameter} -S {stats_dir}')
            else:
                print(f'Skipping {id}...')
    