import argparse
import sys
import os
import re

_MODELS = ['dt', 'lstm', 'transformer']
_DATASETS = ['flows', 'flows15']
_SECTIONS = {
    'losses': {
        'lstm': [],
        'transformer': [],
    },
    'pdp': {
        'lstm': [],
        'transformer': [],
    },
    'neuron': {
        'lstm': []
    },
    'dataset': {
        'flows': [],
        'flows15': []
    }
}

_STRINGS = {
    'lstm': '\gls{lstm}',
    'transformer': 'Transformer model',
    'flows15': 'UNSW-NB15',
    'flows': 'CIC-IDS-2017',
    'flows15_1': '1\% of dataset UNSW-NB15',
    'flows15_10': '10\% of dataset UNSW-NB15',
    'flows15_subset': 'specialized subset UNSW15\_10',
    'flows_1': '1\% of dataset CIC-IDS-2017',
    'flows_10': '10\% of dataset CIC-IDS-2017',
    'flows_subset': 'specialized subset CIC17\_10',
}

_LOSS_TEMPLATE = '''
\\begin{{figure}}[htbp]
	\centering
	\includegraphics[width=1.0\linewidth]{{results/{file}}}
	\caption{{Plot of {type} loss for supervised training on the {model_human} model with {subset_human}.}}
	\label{{fig:appendix:{model}:{subset}_{type}_loss}}
\end{{figure}}
'''

_PDP_TEMPLATE = '''
\\begin{{figure}}[htbp]
	\centering
	\includegraphics[width=1.1\linewidth]{{results/{file}}}
	\caption{{\gls{{pdp}} showing the influence of value variations of feature {feature_human} on classification of {category_human} attacks of the {model_human} pre-trained with proxy tasks as define in \\ref{{{section}}} finetuned with {subset_human}.}}
	\label{{fig:appendix:{model}:pdp:{subset}_{feature}_{category}}}
\end{{figure}}
'''

_NEURON_TEMPLATE = '''
\\begin{{figure}}[htbp]
	\centering
	\includegraphics[width=1.1\linewidth]{{results/{file}}}
	\caption{{Neuron activation plot comparing neuron activations of the latest stage of the \gls{{lstm}} model (after the last packet in the sequence has been processed) after pre-training with the ID proxy task and after fine-tuning with flows of the attack categpry {category_human} of dataset {subset_human}.}}
	\label{{fig:appendix:{model}:neuron:{subset}_{proxy}_{category}}}
\end{{figure}}
'''

_TREE_TEMPALTE = '''
\\begin{{lstlisting}}[captionpos=b, caption={{Decision tree of resulting from a \gls{{dtc}} with depth {depth} fitted on flows of category {category_human} filtered from 90% of dataset {subset_human}. 
The subset constituted of {benign} benign records and {attack} attack records. The resulting validation accuracy was {accuracy}, tested with the remaining 10% of data not used 
for training. }},
label={{lst:appendix:{subset}:tree:{depth}_{category}}}, backgroundcolor=\color{{mygray}}]
	
{tree}

.
\end{{lstlisting}}
'''

_CLEARPAGE_COUNTER = 20

def human(string):
    if string in _STRINGS:
        return _STRINGS[string]
    else:
        return string

def gen_loss_section(apx_file, losses):
    for model, files in losses.items():
        for file in files:
            gen_loss_entry(apx_file, file, model)

def gen_loss_entry(apx_file, file, model):
    head, file_name = os.path.split(file)
    _, subset = os.path.split(head)
    if re.match('[a-z0-9]+_supervised', subset):
        return
    type = re.match(f'([a-z]+)_loss.png', file_name)[1]
    apx_file.write(
        _LOSS_TEMPLATE.format(
            file=file[1:],
            type=type,
            model=model,
            model_human=human(model),
            subset=subset,
            subset_human=human(subset)
        )
    )

def gen_pdp_section(apx_file, pdps):
    for model, files in pdps.items():
        i = 0
        for file in files:
            i += gen_pdp_entry(apx_file, file, model)
            if i % _CLEARPAGE_COUNTER == 0:
                apx_file.write('\n\clearpage\n')
                i += 1

def gen_pdp_entry(apx_file, file, model):
    head, file_name = os.path.split(file)
    _, subset_proxy_tasks = os.path.split(head)
    groups = re.match(f'([a-zA-Z0-9]+)_(\w+)', subset_proxy_tasks)
    subset = groups[1]
    proxy_tasks = groups[2].split('_')
    if(len(proxy_tasks) <= 1):
        return 0
    groups = re.match(f'([a-zA-Z_]+)_([0-9]+)_([a-zA-Z_-]+)_([0-9]+).png', file_name)
    feature_human = groups[1].replace('_',' ')
    feature = groups[2]
    category_human = groups[3].replace('_',' ')
    category = groups[4]
    apx_file.write(
        _PDP_TEMPLATE.format(
            file=file[1:],
            feature=feature,
            feature_human=feature_human,
            category=category,
            category_human=category_human,
            subset=subset,
            subset_human=human(subset + '_subset'),
            model=model,
            model_human=human(model),
            section='sec:experiments:lstm' if model == 'lstm' else 'sec:experiments:transformer'
        )
    )
    return 1

def gen_neuron_section(apx_file, neurons):
    for model, files in neurons.items():
        i = 0
        for file in files:
            i += gen_neuron_entry(apx_file, file, model)
            if i % _CLEARPAGE_COUNTER == 0:
                apx_file.write('\n\clearpage\n')
                i += 1

def gen_neuron_entry(apx_file, file, model):
    file_sanitised = file.replace(' ', '_')
    os.rename(file, file_sanitised)
    head, file_name = os.path.split(file_sanitised)
    head, mode = os.path.split(head)
    if mode == 'means':
        return 0
    _, subset_proxy = os.path.split(head)
    print(subset_proxy)
    groups = re.match(f'([a-zA-Z0-9]+)_(\w+)', subset_proxy)
    subset = groups[1]
    proxy = groups[2]
    groups = re.match(f'([0-9]+)_([a-zA-Z_-]+).png', file_sanitised)
    category = groups[1]
    category_human = groups[2].replace('_',' ')
    apx_file.write(
        _NEURON_TEMPLATE.format(
            file=file_sanitised[1:],
            category=category,
            category_human=category_human,
            subset=subset,
            subset_human=human(subset + '_subset'),
            model=model,
            proxy=proxy
        )
    )
    return 1

def gen_trees_section(apx_file, trees):
    for model, files in trees.items():
        for file in files:
            gen_trees_entry(apx_file, file, model)

def gen_trees_entry(apx_file, file, model):
    file_sanitised = file.replace(' ', '_')
    os.rename(file, file_sanitised)
    head, file_name = os.path.split(file_sanitised)
    head, _ = os.path.split(head)
    _, subset = os.path.split(head)
    groups = re.match(f'dt_[0-9]+_([0-9]+)_([a-zA-Z0-9_-]+).txt', file_name)
    if not groups:
        return
    category = groups[1]
    category_human = groups[2].replace('_',' ')

    with open(file, 'r') as tree_file:
        first_line = tree_file.readline()
        groups = re.match(f'max\. depth\: ([0-9]+), fitting time\: [0-9]+\.[0-9]*s, accuracy\: ([0-9.]+%), ([0-9.]+%) benign samples, ([0-9.]+%) attack samples', first_line)
        depth = groups[1]
        accuracy = groups[2]
        benign = groups[3]
        attack = groups[4]
        tree = ''
        for line in tree_file.readlines():
            tree += line
        apx_file.write(
            _TREE_TEMPALTE.format(
                depth=depth,
                benign=benign,
                attack=attack,
                accuracy=accuracy,
                category=category,
                category_human=category_human,
                subset=subset,
                subset_human=human(subset + '_subset'),
                model=model,
                tree=tree
            )
        )

def main(args):
    for section in _SECTIONS:
        for model in _SECTIONS[section]:
            if section == 'dataset':
                path = f'{args.results_dir}/{section}/{model}/'
            else:
                path = f'{args.results_dir}/{model}/{section}/trees'
            for root, _, files in os.walk(path):
                for file in files:
                    item = os.path.join(root, file)
                    _SECTIONS[section][model].append(item)

    apx_file_path = os.path.join(args.results_dir, args.appendix)
    with open(apx_file_path, 'w+') as apx_file:
        apx_file.write('\n\subsection{Training and Validation Loss}\n')
        print('Generating appendix loss section')
        gen_loss_section(apx_file, _SECTIONS['losses'])

        apx_file.write('\n\subsection{Partial Dependency Plots}\n')
        print('Generating appendix PDP section')
        gen_pdp_section(apx_file, _SECTIONS['pdp'])

        apx_file.write('\n\subsection{Neuron Plots}\n')
        print('Generating appendix neuron plot section')
        gen_neuron_section(apx_file, _SECTIONS['neuron'])

        apx_file.write('\n\subsection{Decision Trees}\n')
        print('Generating appendix decision tree section')
        gen_trees_section(apx_file, _SECTIONS['dataset'])

if __name__=="__main__":
    # Init argument parser
    parser = argparse.ArgumentParser(description='Generate appendix')
    parser.add_argument('-a', '--appendix', help='Appendix file', required=True)
    parser.add_argument('-R', '--results_dir', help='Output directory', required=True)
    args = parser.parse_args(sys.argv[1:])
    main(args)
