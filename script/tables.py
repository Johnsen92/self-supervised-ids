import argparse
import sys
from enum import Enum
import os
import re
import csv
from datetime import datetime

def get_label_from_path(path):
    return re.search(r'\_xy(\w+)', path).group(1)

def stats_table(input_files, output_file):
    out_labels = ['']
    out_rows = []
    first = True
    for file in input_files:
        with open(file, 'r') as f:
            table = csv.reader(f, delimiter=',')
            for i, row in enumerate(table):
                if len(row) == 2 and row[0]=='Proxy task':
                    out_labels.append(row[1])
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

def class_table(input_files, output_file):
    out_labels = ['']
    out_rows = []
    first = True
    for file in input_files:
        if first:
            out_labels.append('')
            out_labels.append('')
        out_labels.append(get_label_from_path(file))
        out_labels.append('')
        with open(file, 'r') as f:
            table = csv.reader(f, delimiter=',')
            for i, row in enumerate(table):
                if len(row) != 5:
                    continue
                if first:
                    out_rows.append(row)
                else:
                    out_rows[i].append(row[-2])
                    out_rows[i].append(row[-1])
            first = False

    with open(output_file, 'w+') as out_f:
        writer = csv.writer(out_f, delimiter=',')
        writer.writerow(out_labels)
        for row in out_rows:
            writer.writerow(row)

class ProxyTask(Enum):
    ALL = 0,
    STATS = 1,
    CLASS = 2,

    def __str__(self):
        return self.name

# Init argument parser
parser = argparse.ArgumentParser(description='Self-seupervised machine learning IDS')
parser.add_argument('-D', '--data_dir', help='Input directory', required=True)
parser.add_argument('-O', '--out_dir', help='Output directory', default='./out')
parser.add_argument('-m', '--mode', default=ProxyTask.ALL, type=lambda proxy_task: ProxyTask[proxy_task], choices=list(ProxyTask), help='Which tables should be generated')
args = parser.parse_args(sys.argv[1:])

regex = {}

if args.mode == ProxyTask.STATS or args.mode == ProxyTask.ALL: 
    regex[ProxyTask.STATS] = r"^stats_"
if args.mode == ProxyTask.CLASS or args.mode == ProxyTask.ALL:
    regex[ProxyTask.CLASS] = r"class\_stats"

data = {}

for k, rx in regex.items():
    data[k] = []
    path = os.walk(args.data_dir)
    for root, directories, files in path:
        for file in files:
            if not re.search(rx, file) is None:
                data[k].append(os.path.join(root, file))

# Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(args.out_dir, exist_ok=True)

out_files = []

for table, file_list in data.items():
    if table == ProxyTask.STATS:
        out_file = args.out_dir + '/' + timestamp + '_stats_comparison_' + str(args.mode) + '.csv'
        stats_table(file_list, out_file)
    elif table == ProxyTask.CLASS:
        out_file = args.out_dir + '/' + timestamp + '_class_comparison_' + str(args.mode) + '.csv'
        class_table(file_list, out_file)
    out_files.append(out_file)

out_files_str = ''
for o_f in out_files:
    out_files_str += o_f + ' '

print(out_files_str[:-1])
#sys.exit(out_files_str[:-2])