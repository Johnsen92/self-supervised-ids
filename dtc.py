i = 0
dtc_table_groups = {}
while True:
    # Props if you can read this line... (just generates a dictionary out of unique entries separated by '|' in groups column)
    with open(args.parameter_file, newline='') as param_file_csv:
        groups = { k:[] for k in list(set([row[CSV_GROUP_INDEX].split('|')[i] for row in [row for row in csv.reader(param_file_csv, delimiter=',', quotechar='"')][2:] if len(row[CSV_GROUP_INDEX].split('|')) > i and row[CSV_GROUP_INDEX].split('|')[i] != ''])) }
        if len(groups.items()) == 0:
            break
        dtc_table_groups.update(groups)
        i += 1

with open(args.string_replace, newline='') as string_replace_file_csv:
    string_replace_list = { row[0]: row[1] for row in csv.reader(string_replace_file_csv, delimiter=',', quotechar='"') if len(row) == 2 }

ids = []
param_file_csv = open(args.parameter_file, newline='')
rows = [row for row in csv.reader(param_file_csv, delimiter=',', quotechar='"')]
datasets = []
for i, row in enumerate(rows):
    if i == 1:
        parameters = row[3:]
    elif i == 0:
        labels = row
    elif row[CSV_MODEL_INDEX] == args.model:
        parameter_string, parameter_list = build_parameters(parameters, row[3:])
        model_parser = DTArgumentParser(parameter_list)
        train = train_trans
        base_dir = f'{args.stats_dir}/{row[CSV_MODEL_INDEX]}'
        stats_dir = f'{base_dir}/stats/'
        log_dir = f'{base_dir}/runs/'
        errorlog_dir = f'{args.stats_dir}/logs/'
        make_dir(errorlog_dir)
        add_parameter(parameter_list, '-S', stats_dir)
        add_parameter(parameter_list, '-L', log_dir)
        if args.debug:
            add_parameter(parameter_list, '--debug')
        model_args = model_parser.parse_args(parameter_list)
        if len(args.proxy_tasks) != 0 and not model_args.proxy_task in args.proxy_tasks:
            continue
        add_parameter(parameter_list, '--id_only')
        model_args_id_only = model_parser.parse_args(parameter_list)
        id = train(model_args_id_only)
        experiment = row[CSV_EXPERIMENT_INDEX]
        ids.append(id)
        datasets.append((model_args.data_file, model_args.benign_category))
        for group in row[CSV_GROUP_INDEX].split('|'): 
            if group != '':
                dtc_table_groups[group].append((id, experiment))
        stats_dir_extended = f'{stats_dir}{id}/'
        print(f'----------------------------------------------------------------{i-1}/{len(rows)-2}---------------------------------------------------------------------------')
        try:
            if not os.path.exists(stats_dir_extended):
                print(f'Make results for ID {id}...')
                train(model_args)
            elif len(os.listdir(stats_dir_extended)) != EXPECTED_RESULTS_FILES:
                print(f'Make results for ID {id}...')
                os.system(f'rm -r "{stats_dir_extended}" -f')
                train(model_args)
            else:
                print(f'Skipping {id}...')
        except not KeyboardInterrupt:
            with open(f'{errorlog_dir}/{datetime.now():%Y%m%d}_{id}.txt','w+') as f:
                f.write(traceback.format_exc())
                f.write(f'Error in run {id}')
            print(f'An error occured during run {id}')

# Generate CSV Tables
table_dir = f'{base_dir}/tables/'
rm_dir(table_dir)
for k, group_entries in dtc_table_groups.items():
    if len(group_entries) <= 1:
        continue
    group_dir = f'{table_dir}/{k}'
    print(f'Generating tables for group {k}...', end='')
    generate_tables(group_entries, Mode.ALL, stats_dir, group_dir, order=ids)
    print('done.')

