import sys
from torch.utils.data import random_split, DataLoader
from torch import optim, nn
from classes import datasets, statistics, utils, transformer, trainer
from classes.datasets import Flows, FlowsSubset
import torch
import os.path
import jsons
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import random

def main(args):
    # If val_epochs is set to auto mode, calculate reasonable value
    if args.val_epochs == -1:
        args.val_epochs = max(1, args.n_epochs // 100)

    # Set random seed
    if args.random_seed == 0:
        SEED = random.randint(1, pow(2,16)-1)
    else:
        SEED = args.random_seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random_seed = SEED

    # Serialize arguments and store them in json export folder
    with open(args.json_dir + '/args.json', 'w') as f:
        f.write(jsons.dumps(args))

    # Datafile basename
    data_filename = os.path.basename(args.data_file)[:-7]

    # Identifier for current parameters
    run_id = f'transformer_{data_filename}_rn{random_seed}_do{str(args.dropout*10).replace(".", "")}_nl{args.n_layers}_nh{args.n_heads}_fx{args.forward_expansion}_bs{args.batch_size}_lr{str(args.learning_rate*10).replace(".", "")}'
        
    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pretraining ID
    pretraining_id = f'sp{args.self_supervised}_sep{args.n_epochs_pretraining}_xy{args.proxy_task}'

    # Training ID
    training_id = f'tep{args.n_epochs}_tp{args.train_percent}'

    if not args.subset_config is None:
        training_id += '_subset|' + os.path.basename(args.subset_config)[:-5]
    if args.debug:
        training_id += '_debug'

    # ID and unique ID (with timestamp) for this run
    id = f'{run_id}_{training_id}_{pretraining_id}'
    uid = f'{timestamp}_{id}'

    if args.id_only:
        print(id)
        return id

    # Init cache
    general_cache = utils.Cache(cache_dir=args.cache_dir, key_prefix=id, disabled=args.no_cache, label='Transformer Cache')
    training_cache = utils.Cache(cache_dir=args.cache_dir + '/training', key_prefix=id, disabled=args.no_cache, label='Transformer Training Cache')
    pretraining_cache = utils.Cache(cache_dir=args.cache_dir + '/pretraining', key_prefix=run_id + '_' + pretraining_id, disabled=args.no_cache, label='Transformer Pretraining Cache')

    # Extended stats directory for this run
    extended_stats_dir = (args.stats_dir if args.stats_dir[-1] == '/' else args.stats_dir + '/') + id + '/'

    # Load dataset and normalize data, or load from cache
    cache_filename = f'dataset_normalized_{data_filename}'
    if not general_cache.exists(cache_filename, no_prefix=True):
        dataset = Flows(data_pickle=args.data_file, cache=general_cache, max_length=args.max_sequence_length, remove_changeable=args.remove_changeable)
        #dataset = FlowsSubset(dataset_all, dataset_all.mapping, min_flow_length=args.min_sequence_length)
        general_cache.save(cache_filename, dataset, no_prefix=True, msg='Storing normalized dataset')
    else:
        dataset = general_cache.load(cache_filename, no_prefix=True, msg='Loading normalized dataset')

    # Get category mapping from dataset 
    category_mapping = dataset.mapping

    # Number of samples
    n_samples = len(dataset)

    # Won't get far without GPU, so I assume you have one...
    device = torch.device('cuda:0')

    # Split dataset into training and validation parts
    validation_size = int(round((n_samples * args.val_percent) / 1000.0))
    supervised_size = int(round((n_samples * args.train_percent) / 1000.0))
    pretraining_size = int(round((n_samples * args.self_supervised) / 1000.0))

    # correct for rounding error
    if (validation_size + supervised_size + pretraining_size) - n_samples == 1:
        validation_size -= 1

    # If debug flag is set, use exactly one batch for pretraining, training and validation
    if args.debug:
        validation_size = supervised_size = pretraining_size = args.batch_size
        args.n_epochs = 1
        args.n_epochs_pretraining = 1

    # Split dataset into pretraining, training and validation set
    if args.self_supervised > 0:
        train_data, pretrain_data, val_data = dataset.split([supervised_size, pretraining_size, validation_size], stratify=True)
    else:
        train_data, val_data = dataset.split([supervised_size, validation_size], stratify=True)

    # If the subset flag is set, only use this small selected dataset for supervised learning
    if not args.subset_config is None:
        train_data = FlowsSubset(train_data, category_mapping, config_file=args.subset_config, key='TRAIN', config_index=args.subset_config_index)
        val_data = FlowsSubset(val_data, category_mapping, config_file=args.subset_config, key='VALIDATE', config_index=args.subset_config_index)
        if args.self_supervised > 0:
            pretrain_data = FlowsSubset(pretrain_data, category_mapping, config_file=args.subset_config, key='PRETRAIN', config_index=args.subset_config_index)

    # Assure val batch size is at most the size of validation data split
    args.val_batch_size = min(len(val_data), args.val_batch_size)

    # Init data loaders
    if args.self_supervised > 0:
        pretrain_loader = DataLoader(dataset=pretrain_data, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows, drop_last=True)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows, drop_last=True)
    if args.debug:
        test_loader = val_loader
    else:
        test_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, collate_fn=datasets.collate_flows, drop_last=True)

    # Define input and output data width
    data, _, _ = dataset[0]
    input_size = data.size()[1]
    output_size = args.output_size

    # Init model
    transformer_model = transformer.Transformer(
        input_size = input_size,
        num_heads = args.n_heads,
        num_encoder_layers = args.n_layers,
        num_decoder_layers = args.n_layers,
        forward_expansion = input_size * args.forward_expansion,
        dropout = args.dropout,
        max_len = args.max_sequence_length
    ).to(device)

    # Init transformer encoder
    model = transformer.TransformerEncoder(
        encoder = transformer_model.transformer.encoder,
        input_size = input_size,
        output_size = args.output_size,
        dropout = args.dropout,
        max_len = args.max_sequence_length
    ).to(device)

    # Init optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Gather model parameters for statistics
    model_parameters = {
        '# Layers' : args.n_layers,
        '# Heads' : args.n_heads,
        'Forward expansion' : args.forward_expansion,
        'Dropout' : args.dropout
    }

    # If pretraining epochs argument is set, use it, else default to supervised epochs argument
    epochs_pretraining = args.n_epochs if args.n_epochs_pretraining == 0 else args.n_epochs_pretraining

    # Init statistics object
    pretraining_stats = statistics.Stats(
        stats_dir = extended_stats_dir,
        benign = args.benign_category,
        category_mapping = category_mapping,
        proxy_task = f'{args.proxy_task}',
        pretrain_percent = args.self_supervised,
        train_percent = args.train_percent,
        val_percent = args.val_percent,
        n_epochs = args.n_epochs,
        n_epochs_pretraining = epochs_pretraining,
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        model_parameters = model_parameters,
        random_seed = random_seed,
        subset = ''
    )

    training_stats = statistics.Stats(
        stats_dir = extended_stats_dir,
        benign = args.benign_category,
        category_mapping = category_mapping,
        proxy_task = f'{args.proxy_task}',
        pretrain_percent = args.self_supervised,
        train_percent = args.train_percent,
        val_percent = args.val_percent,
        n_epochs = args.n_epochs,
        n_epochs_pretraining = epochs_pretraining,
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        model_parameters = model_parameters,
        random_seed = random_seed,
        subset = os.path.basename(args.subset_config)[:-5] if not args.subset_config is None else ''
    )

    # Init summary writer for TensorBoard
    writer = SummaryWriter(f'{args.log_dir}/{uid}')

    # Pretraining if enabled
    if args.self_supervised > 0:
        # Init pretraining criterion
        pretraining_criterion = nn.L1Loss()
        if(args.proxy_task == utils.ProxyTask.INTER):
            trainer.Transformer.Interpolation(
                model = model, 
                training_data = pretrain_loader, 
                device = device, 
                criterion = pretraining_criterion, 
                optimizer = optimizer, 
                epochs = epochs_pretraining, 
                val_epochs = args.val_epochs,
                stats = pretraining_stats, 
                cache = pretraining_cache,
                json = args.json_dir,
                writer = writer,
                title = 'Interpolation',
                test_data = test_loader
            )
        elif(args.proxy_task == utils.ProxyTask.AUTO):
            # Introduce dropout for denoising autoencoder
            #model.dropout = nn.Dropout(0.2)
            pretrainer = trainer.Transformer.AutoEncode(
                model = model, 
                training_data = pretrain_loader, 
                device = device, 
                criterion = pretraining_criterion, 
                optimizer = optimizer, 
                epochs = epochs_pretraining, 
                val_epochs = args.val_epochs,
                stats = pretraining_stats, 
                cache = pretraining_cache,
                json = args.json_dir,
                writer = writer,
                title = 'AutoEncoder',
                test_data = test_loader
            )
        elif(args.proxy_task == utils.ProxyTask.OBSCURE):
            pretrainer = trainer.Transformer.ObscureFeature(
                model = model, 
                training_data = pretrain_loader, 
                device = device, 
                criterion = pretraining_criterion, 
                optimizer = optimizer, 
                epochs = epochs_pretraining, 
                val_epochs = args.val_epochs,
                stats = pretraining_stats, 
                cache = pretraining_cache,
                json = args.json_dir,
                writer = writer,
                title = 'ObscureFeature',
                test_data = test_loader
            )
        elif(args.proxy_task == utils.ProxyTask.MASK):
            pretrainer = trainer.Transformer.MaskPacket(
                model = model, 
                training_data = pretrain_loader, 
                device = device, 
                criterion = pretraining_criterion, 
                optimizer = optimizer, 
                epochs = epochs_pretraining, 
                val_epochs = args.val_epochs,
                stats = pretraining_stats, 
                cache = pretraining_cache,
                json = args.json_dir,
                writer = writer,
                title = 'MaskPacket',
                test_data = test_loader
            )
        else:
            print(f'Proxy task can not be {args.proxy_task} for self supervised training')

        # Pretrain
        pretrainer.train()

        # If config file is provided, make neuron activation data
        if not args.neuron_config is None:
            pretrainer.neuron_activation(id, args.neuron_config, postfix='pre', title='Pretraining', batch_first=False)

        if args.proxy_task == utils.ProxyTask.AUTO:
            model.dropout = nn.Dropout(args.dropout)      


    # Init training criterion
    training_criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # Switch model into supervised fine-tuning mode
    model.tune()

    # Init trainer for supervised training
    finetuner = trainer.Transformer.Supervised(
        model = model, 
        training_data = train_loader, 
        validation_data = val_loader,
        device = device, 
        criterion = training_criterion, 
        optimizer = optimizer, 
        epochs = args.n_epochs, 
        val_epochs = args.val_epochs,
        stats = training_stats, 
        cache = training_cache,
        json = args.json_dir,
        writer = writer,
        title = 'Supervised',
        test_data = test_loader
    )

    # Train model
    finetuner.train()

    # Partial dependency data
    if args.proxy_task == utils.ProxyTask.NONE and not args.pdp_config is None:
        finetuner.pdp(id, args.pdp_config, batch_first=False)

    # Neuron activation data
    if not args.neuron_config is None:
        finetuner.neuron_activation(id, args.neuron_config, postfix='Supervised', batch_first=False)

    # Remove temp directories
    general_cache.clean()
    training_cache.clean()
    pretraining_cache.clean()

    # Print and save stats
    if not args.debug:
        finetuner.evaluate()

    return True

if __name__=="__main__":
    # Init argument parser
    parser = utils.TransformerArgumentParser(description='Self-seupervised machine learning IDS')
    args = parser.parse_args(sys.argv[1:])