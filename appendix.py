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
    'flows15_1': '1\% of dataset UNSW-NB15',
    'flows15_10': '10\% of dataset UNSW-NB15',
    'flows15_subset': 'specialized subset UNSW15\_10',
    'flows_1': '1\% of dataset CIC-IDS-2017',
    'flows_10': '10\% of dataset CIC-IDS-2017',
    'flows_subset': 'specialized subset CIC17\_10',
}

_LOSS_TEMPLATE = '''
\\begin{{figure}}[!htbp]
	\centering
	\includegraphics[width=1.0\linewidth]{{results/{file}}}
	\caption{{Plot of {type} loss for supervised training on the {model_human} model with {subset_human}.}}
	\label{{fig:results:{model}:{subset}_{type}_loss}}
\end{{figure}}
'''

_PDP_TEMPLATE = '''
\begin{{figure}}[!htbp]
	\centering
	\includegraphics[width=1.1\linewidth]{{results/{file}}}
	\caption{{Partial dependency plot showing the influence of value variations of feature {feature_human} on classification of {category_human} attacks in the {dataset_human} dataset.}}
	\label{{fig:results:{model}:pdp:{subset}_{feature}_{category}}}
\end{{figure}}
'''

_NEURON_TEMPLATE = '''
'''

_TREE_TEMPALTE = '''
'''

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
        for file in files:
            gen_loss_entry(apx_file, file, model)

def gen_pdp_entry(apx_file, file, model):
    head, file_name = os.path.split(file)
    _, subset = os.path.split(head)
    type = re.match(f'([a-z]+)_loss.png', file_name)[1]

    apx_file.write(
        _PDP_TEMPLATE.format(
            file=file[1:],
            feature=feature,
            feature_human=feature_human,
            category=category,
            category_humand=category_human,
            dataset_human=dataset_human,
            model=model,
            subset=subset,
            training_data=human(subset)
        )
    )

def main(args):
    for section in _SECTIONS:
        for model in _SECTIONS[section]:
            if section == 'dataset':
                path = f'{args.results_dir}/{section}/{model}/'
            else:
                path = f'{args.results_dir}/{model}/plots/{section}/'
            for root, _, files in os.walk(path):
                for file in files:
                    item = os.path.join(root, file)
                    _SECTIONS[section][model].append(item)

    apx_file_path = os.path.join(args.results_dir, args.appendix)
    with open(apx_file_path, 'w+') as apx_file:
        apx_file.write('\subsection{Training and Validation Loss}')
        gen_loss_section(apx_file, _SECTIONS['losses'])
        apx_file.write('\subsection{Partial Dependency Plots}')
        apx_file.write('\subsection{Neuron Plots}')
        apx_file.write('\subsection{Decision Trees}')

if __name__=="__main__":
    # Init argument parser
    parser = argparse.ArgumentParser(description='Generate appendix')
    parser.add_argument('-a', '--appendix', help='Appendix file', required=True)
    parser.add_argument('-R', '--results_dir', help='Output directory', required=True)
    args = parser.parse_args(sys.argv[1:])
    main(args)
