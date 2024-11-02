import argparse
import os
import json
import torch
import time
import utils
import torchvision
import torch.nn as nn
from marshmallow import Schema, fields
from schemas import ModelEMAConfigSchema, EarlyStopperConfigSchema, DataLoaderConfigSchema, registered_schemas, registered_schemas_types
from sklearn.model_selection import ParameterGrid
from trainer_tester import SUPPORTED_MODELS, train_loop, test, gen_default_transforms_config

def get_args_parser():
    parser = argparse.ArgumentParser(
        description='-= Automated PyTorch DCNN training and testing script -=', 
        add_help=True,
        )
    
    parser.add_argument(
        'dataset_path',
        metavar='dataset-path',
        type=str,
        help='Path to image dataset, structured like ImageNet',
        )
    
    parser.add_argument(
        'model',
        type=str, 
        choices=SUPPORTED_MODELS,
        help='DCNN model to train',
        )
    
    parser.add_argument(
        '--model-search', 
        type=str, 
        default=None,
        help='Path to JSON configuration file defining parameter search space for model',
        )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=90,
        help='Maximum number of training iterations for each model',
        )
    
    parser.add_argument(
        '--optimizer', 
        type=str, 
        choices=utils.SUPPORTED_OPTIMIZERS, 
        default='adam', 
        help='Optimizer to use for each model',
        )
    
    parser.add_argument(
        '--optimizer-search', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file defining parameter search space for optimizer',
        )
    
    parser.add_argument(
        '--lr-search', 
        type=float, 
        nargs='+',
        default=[0.1], 
        help='Learning rate search space to use for optimizer, learning rate scheduler, etc.',
        )
    
    parser.add_argument(
        '--lr-scheduler', 
        type=str, 
        choices=utils.SUPPORTED_LR_SCHEDULERS, 
        default=None, 
        help='Learning rate scheduler to use for each model. If unspecified, no learning rate scheduler is used.',
        )
    
    parser.add_argument(
        '--lr-scheduler-search', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file defining parameter search space for learning rate scheduler',
        )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda', 
        help='Hardware device to use for training/inference',
        )
    
    parser.add_argument(
        '--val-ratio', 
        type=float, 
        default=0.2, 
        help='Ratio of data to use for model validation',
        )
    
    parser.add_argument(
        '--test-ratio', 
        type=float, 
        default=0.2, 
        help='Ratio of data to use for model testing',
        )
    
    parser.add_argument(
        '--train-transforms-config', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for train dataset transforms',
        )

    parser.add_argument(
        '--val-transforms-config', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for validation dataset transforms',
        )
    
    parser.add_argument(
        '--test-transforms-config', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for test dataset transforms',
        )
    
    parser.add_argument(
        '--train-dataloader-search', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file defining parameter search space for train dataloader',
        )
    
    parser.add_argument(
        '--val-dataloader-search', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file defining parameter search space for validation dataloader',
        )
    
    parser.add_argument(
        '--test-dataloader-search', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file defining parameter search space for test dataloader',
        )
    
    parser.add_argument(
        '--model-ema-search', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file defining parameter search space for EMA of model parameters. If unspecified, EMA will not be applied to model parameters.',
        )
    
    parser.add_argument(
        '--early-stopper-search', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file defining parameter search space for early stopper. If unspecified, no early stopping is performed.',
        )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Seed to use for controlling randomness throughout training/testing',
        )
    
    parser.add_argument(
        '--averaging-method', 
        type=str, 
        choices=['micro','macro',], 
        default='micro', 
        help='Averaging method to use when calculating metrics',
        )
    
    parser.add_argument(
        '--plot-metrics', 
        action='store_true', 
        help='If specified, generate plots of metrics for each model',
        )
            
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None, 
        help='Filepath of directory to save model checkpoints, metrics, and related outputs to',
        )
    
    return parser

class AutoParamGridSampleSchema():
    def __init__(self, optim_name:str, lr_sched_name:str|None=None, incl_model_ema=False, incl_early_stopper=False):
        self.optim_name = optim_name
        self.lr_sched_name = lr_sched_name
        self.incl_model_ema = incl_model_ema
        self.incl_early_stopper = incl_early_stopper
        self.schema = self._make_schema()

    def _make_schema(self):
        '''
        Represents a sample in a parameter grid 
        used during automated training/testing of DCNNs

        Each sample is a dict mapping config names for various objects (model, optimizer, etc.)
        to a dict mapping hyperparamter names to values

        Ex: {'model':{'dropout':0.5}, 'optimizer':{'lr':0.1}, ...}
        '''

        schema_dict = {
            'model_config':fields.Dict(keys=fields.Str, required=True),
            'train_dataloader_config':fields.Nested(DataLoaderConfigSchema, required=True),
            'val_dataloader_config':fields.Nested(DataLoaderConfigSchema, required=True),
            'test_dataloader_config':fields.Nested(DataLoaderConfigSchema, required=True),
            }
        
        for schema_name in registered_schemas_types:
            if registered_schemas_types[schema_name] == 'optimizer' and schema_name.lower() == utils.SUPPORTED_OPTIMIZERS[self.optim_name].__name__.lower():
                schema_dict['optimizer_config'] = fields.Nested(registered_schemas[schema_name], required=True)
            elif registered_schemas_types[schema_name] == 'lr_scheduler' and self.lr_sched_name and schema_name.lower() == utils.SUPPORTED_LR_SCHEDULERS[self.lr_sched_name].__name__.lower():
                schema_dict['lr_scheduler_config'] = fields.Nested(registered_schemas[schema_name], required=True)
        if 'optimizer_config' not in schema_dict:
            raise Exception(f'No such optimizer schema exists for {self.optim_name} optimizer')
        if self.lr_sched_name and 'lr_scheduler_config' not in schema_dict:
            raise Exception(f'No such learning rate scheduler exists for {self.lr_sched_name} learning rate scheduler')
            
        if self.incl_model_ema:
            schema_dict['model_ema_config'] = fields.Nested(ModelEMAConfigSchema, required=True)
        
        if self.incl_early_stopper:
            schema_dict['early_stopper_config'] = fields.Nested(EarlyStopperConfigSchema, required=True)

        return Schema.from_dict(schema_dict)()

    def validate(self, sample:dict):
        return self.schema.load(sample)
    
def gen_default_dataloader_search_space():
    return {
        'batch_size':[32],
        'shuffle':[False],
        'num_workers':[0],
        'pin_memory':[True],
        }

def make_param_grid(search_spaces:dict):
    param_grid = {}

    # maps an unpacked param name (search space name + separator + param name) to a tuple of the search space and original param name
    unpacked_param_names = {}

    # unpack each search space dict
    for search_space_name in search_spaces:
        search_space = search_spaces[search_space_name]
        for param in search_space:
            param_name = f'{search_space_name}_{param}'
            param_grid[param_name] = search_space[param]
            unpacked_param_names[param_name] = (search_space_name, param)
    
    # generate all possible combinations of params
    grid = list(ParameterGrid(param_grid))

    # prepare new grid, where each sample has its params repacked into their associated dicts
    refined_grid = [{search_space_name:{} for search_space_name in search_spaces} for _ in range(len(grid))]

    # repack params back into their associated dicts
    for i in range(len(grid)):
        sample = grid[i]
        for param in sample:
            search_space_name, param_name = unpacked_param_names[param]
            refined_grid[i][search_space_name][param_name] = sample[param]

    return refined_grid

def grid_loop(train_dataset, val_dataset, test_dataset, param_grid, model_name:str, optim_name:str, epochs:int, device, lr_sched_name:str='', averaging_method='micro', num_classes:int|None=None, plot_metrics=False, output_dir=''):
    for i, sample in enumerate(param_grid):
        print(f'Trying sample {i+1} of {len(param_grid)} hyperparameter samples...')

        # validate sample contains correct keys and valid values
        sample_schema = AutoParamGridSampleSchema(
            optim_name, 
            lr_sched_name=lr_sched_name, 
            incl_model_ema='model_ema_config' in sample and len(sample['model_ema_config']) > 0, 
            incl_early_stopper='early_stopper_config' in sample,
            )
        sample = sample_schema.validate(sample)

        print(f'Making dataloaders...')
        train_dataloader = utils.make_dataloader(
            train_dataset, 
            sample['train_dataloader_config'],
            )
        val_dataloader = utils.make_dataloader(
            val_dataset,
            sample['val_dataloader_config']
            )
        test_dataloader = utils.make_dataloader(
            test_dataset,
            sample['test_dataloader_config'],
            )
        print(f'Samples in train dataset: {len(train_dataloader.dataset)}')
        print(f'Samples in val dataset: {len(val_dataloader.dataset)}')
        print(f'Samples in test dataset: {len(test_dataloader.dataset)}')

        print(f'Creating {model_name} model...')
        model = torchvision.models.get_model(model_name, **sample['model_config'])
        model_ema = None
        if 'model_ema_config' in sample:
            print('Creating model EMA...')
            model_ema = utils.make_model_ema(model, model_ema_config=sample['model_ema_config'])
            model_ema.to(device)
        model.to(device)

        print(f'Creating {optim_name} optimizer...')
        optimizer = utils.make_optimizer(
            model.parameters(), 
            optim_name, 
            optimizer_config=sample['optimizer_config'],
            )

        print('Creating loss function...')
        criterion = nn.CrossEntropyLoss()

        lr_scheduler = None
        if lr_sched_name:
            print(f'Creating {lr_sched_name} learning rate scheduler...')
            lr_scheduler = utils.make_lr_scheduler(
                optimizer, 
                lr_sched_name, 
                sample['lr_scheduler_config'],
                )
            
        early_stopper = None
        if 'early_stopper_config' in sample:
            print('Creating early stopper...')
            early_stopper = utils.make_early_stopper(sample['early_stopper_config'])

        # save jere any checkpoints/metrics for model trained on this sample of hyperparameters
        trial_dir_path = f'trial{i+1}'
        if output_dir:
            trial_dir_path = os.path.join(output_dir, trial_dir_path)
        os.makedirs(trial_dir_path, exist_ok=True)

        # save hyperparameters sample
        with open(os.path.join(trial_dir_path, 'hyperparams.json'), 'w') as f:
            json.dump(sample, f, indent=4)

        print('Training...')
        train_metrics, val_metrics = train_loop(
            epochs, 
            model, 
            train_dataloader, 
            val_dataloader, 
            criterion, 
            optimizer, 
            device, 
            lr_scheduler=lr_scheduler,
            model_ema=model_ema,
            model_ema_steps=sample['model_ema_config']['steps'] if 'model_ema_config' in sample and 'steps' in sample['model_ema_config'] else 1,
            early_stopper=early_stopper,
            averaging_method=averaging_method,
            num_classes=num_classes,
            output_dir=trial_dir_path,
            )
        
        print('Testing...')
        test_metrics = test(
            model, 
            test_dataloader, 
            device, 
            averaging_method=averaging_method, 
            num_classes=num_classes,
            )
        
        print('Saving metrics...')
        metrics_dir_path = 'metrics'
        if output_dir:
            metrics_dir_path = os.path.join(output_dir, metrics_dir_path)
        os.makedirs(metrics_dir_path, exist_ok=True)
        curr_time = int(time.time())
        train_metrics.to_csv(os.path.join(metrics_dir_path, f'train_metrics_{curr_time}.csv'), index=False)
        val_metrics.to_csv(os.path.join(metrics_dir_path, f'val_metrics_{curr_time}.csv'), index=False)
        test_metrics.to_csv(os.path.join(metrics_dir_path, f'test_metrics_{curr_time}.csv'), index=False)
        print(f'Test metrics: \n{test_metrics.to_string(index=False)}')

        if plot_metrics:
            print('Plotting metrics comparisons...')
            utils.plot_comparison(
                [i for i in range(len(train_metrics))], # we assume train_metrics and val_metrics will have same length
                train_metrics['acc'], 
                val_metrics['acc'], 
                label1='train_acc', 
                label2='val_acc', 
                title='Training & Validation Accuracy',
                x_label='Epoch',
                y_label='Accuracy',
                filename='acc_comparison', 
                output_dir=metrics_dir_path,
                )
            utils.plot_comparison(
                [i for i in range(len(train_metrics))], 
                train_metrics['loss'], 
                val_metrics['loss'], 
                label1='train_loss', 
                label2='val_loss', 
                title='Training & Validation Loss', 
                x_label='Epoch',
                y_label='Loss',
                filename='loss_comparison', 
                output_dir=metrics_dir_path,
                )

def main(args:argparse.Namespace):
    # save command line args to disk
    cmd_args_output_path = 'cmd_args.json'
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        cmd_args_output_path = os.path.join(args.output_dir, cmd_args_output_path)
    with open(cmd_args_output_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device(args.device)

    print('Making dataset transforms...')
    # transforms are not considered part of the search space
    train_dataset_transforms = utils.make_transforms_pipeline(
        gen_default_transforms_config() if args.train_transforms_config is None else json.load(open(args.train_transforms_config)),
        )
    val_dataset_transforms = utils.make_transforms_pipeline(
        gen_default_transforms_config() if args.val_transforms_config is None else json.load(open(args.val_transforms_config)),
        )
    test_dataset_transforms = utils.make_transforms_pipeline(
        gen_default_transforms_config() if args.test_transforms_config is None else json.load(open(args.test_transforms_config)),
        )

    print('Loading dataset...')
    full_dataset = utils.load_dataset(args.dataset_path)
    train_dataset, val_dataset, test_dataset = utils.split_dataset(
        full_dataset, 
        train_ratio=1.0-(args.val_ratio+args.test_ratio), 
        val_ratio=args.val_ratio, 
        test_ratio=args.test_ratio, 
        seed=args.seed,
        )
    train_dataset.dataset.transform = train_dataset_transforms
    val_dataset.dataset.transform = val_dataset_transforms
    test_dataset.dataset.transform = test_dataset_transforms

    num_classes = len(full_dataset.classes)
    del full_dataset
    print(f'Dataset has {num_classes} classes')

    # load any user-supplied search spaces from args
    print('Loading search spaces...')
    search_spaces = {}

    model_search_space= {}
    if args.model_search is not None:
        print(f'Loading search space for {args.model} model...')
        model_search_space = json.load(open(args.model_search))
    model_search_space['num_classes'] = [num_classes]
    search_spaces['model_config'] = model_search_space

    model_ema_search_space = {}
    if args.model_ema_search is not None:
        print('Loading model EMA search space...')
        model_ema_search_space = json.load(open(args.model_ema_search))
        search_spaces['model_ema_config'] = model_ema_search_space

    optim_search_space = {}
    if args.optimizer_search is not None:
        print(f'Loading search space for {args.optimizer} optimizer...')
        optim_search_space = json.load(open(args.optimizer_search))
    optim_search_space['lr'] = args.lr_search # only the optimizer needs the learning rate(s) directly
    search_spaces['optimizer_config'] = optim_search_space

    lr_sched_search_space = {}
    if args.lr_scheduler is not None:
        print(f'Loading search space for {args.lr_scheduler} learning rate scheduler...')
        if args.lr_scheduler_search is None:
            raise Exception('Must supply a learning rate scheduler search space if a learning rate scheduler is specified')
        lr_sched_search_space = json.load(open(args.lr_scheduler_search))
        search_spaces['lr_scheduler_config'] = lr_sched_search_space

    print('Loading search spaces for dataloaders...')
    train_dl_search_space = gen_default_dataloader_search_space()
    if args.train_dataloader_search is not None:
        train_dl_search_space = json.load(open(args.train_dataloader_search))
    val_dl_search_space = gen_default_dataloader_search_space()
    if args.val_dataloader_search is not None:
        val_dl_search_space = json.load(open(args.val_dataloader_search))
    test_dl_search_space = gen_default_dataloader_search_space()
    if args.test_dataloader_search is not None:
        test_dl_search_space = json.load(open(args.test_dataloader_search))
    search_spaces['train_dataloader_config'] = train_dl_search_space
    search_spaces['val_dataloader_config'] = val_dl_search_space
    search_spaces['test_dataloader_config'] = test_dl_search_space

    early_stopper_search_space = {}
    if args.early_stopper_search is not None:
        print('Loading search space for early stopper...')
        early_stopper_search_space = json.load(open(args.early_stopper_search))
        search_spaces['early_stopper_config'] = early_stopper_search_space

    print('Making parameter grid...')
    param_grid = make_param_grid(search_spaces)
    
    print('Performing grid search over hyperparameters...')
    grid_loop(
        train_dataset, 
        val_dataset, 
        test_dataset, 
        param_grid, 
        args.model, 
        args.optimizer, 
        args.epochs, 
        device, 
        lr_sched_name=args.lr_scheduler, 
        averaging_method=args.averaging_method, 
        num_classes=num_classes, 
        output_dir=args.output_dir,
        )

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)