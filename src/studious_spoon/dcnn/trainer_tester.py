import os
import json
import time
import torch
import utils
import argparse
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from termcolor import colored, cprint

SUPPORTED_MODELS = {
    'vgg16','vgg19','alexnet','googlenet',
    'resnet18','efficientnet_b0','efficientnet_b1',
    'efficientnet_b2','efficientnet_b3','efficientnet_b4',
    'efficientnet_b5','efficientnet_b6','efficientnet_b7',
    }

def get_args_parser():
    parser = argparse.ArgumentParser(
        description='-= PyTorch DCNN training and testing script -=', 
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
        '--model-config', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for model',
        )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=90,
        help='Maximum number of training iterations for model',
        )
    
    parser.add_argument(
        '--optimizer', 
        type=str, 
        choices=utils.SUPPORTED_OPTIMIZERS, 
        default='adam', 
        help='Optimizer to use',
        )
    
    parser.add_argument(
        '--optimizer-config', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for optimizer',
        )
    
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.1, 
        help='Learning rate to use for optimizer, learning rate scheduler, etc.',
        )
    
    parser.add_argument(
        '--lr-scheduler', 
        type=str, 
        choices=utils.SUPPORTED_LR_SCHEDULERS, 
        default=None, 
        help='Learning rate scheduler to use. If unspecified, no learning rate scheduler is used.',
        )
    
    parser.add_argument(
        '--lr-scheduler-config', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for learning rate scheduler',
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
        '--train-dataloader-config', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for train dataloader',
        )
    
    parser.add_argument(
        '--val-dataloader-config', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for validation dataloader',
        )
    
    parser.add_argument(
        '--test-dataloader-config', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for test dataloader',
        )
    
    parser.add_argument(
        '--model-ema-config', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for EMA of model parameters. If unspecified, EMA will not be applied to model parameters.',
        )
    
    parser.add_argument(
        '--early-stopper-config', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for early stopper. If unspecified, no early stopping is performed.',
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
        help='If specified, generate plots of metrics',
        )
            
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None, 
        help='Filepath of directory to save model checkpoints, metrics, and related outputs to',
        )
    
    return parser

def gen_default_dataloader_config():
    return {
        'batch_size':32,
        'shuffle':False,
        'num_workers':0,
        'pin_memory':True,
        }

def gen_default_transforms_config():
    return {
        'Resize':{'size':(224,224),},
        'ToTensor':{},
        'Normalize':{'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225],},
        }

def train_step(model, train_dataloader, optimizer, criterion, device, model_ema=None, model_ema_steps=1, averaging_method='micro', num_classes:int|None=None):
    model.train()

    step_metrics = {'acc':[], 'recall':[], 'precision':[], 'f1':[], 'loss':[]}

    for i, (samples, targets) in tqdm(enumerate(train_dataloader), desc='Training Step'):
        samples, targets = samples.to(device), targets.to(device)

        output = model(samples)

        if isinstance(model, torchvision.models.GoogLeNet):
            logits = output[0]
            aux_logits2 = output[1]
            aux_logits1 = output[2]

            logits_loss = criterion(logits, targets)
            aux_logits1_loss = criterion(aux_logits1, targets)
            aux_logits2_loss = criterion(aux_logits2, targets)
            loss = logits_loss + (0.3 * aux_logits1_loss) + (0.3 * aux_logits2_loss)
        else:
            loss = criterion(output, targets)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if model_ema and i % model_ema_steps == 0:
            model_ema.update_parameters(model)

        if isinstance(model, torchvision.models.GoogLeNet):
             _, preds = torch.max(output[0], 1)
        else:
            _, preds = torch.max(output, 1)
        batch_metrics = utils.calc_metrics(preds, targets, average=averaging_method, num_classes=num_classes)
        batch_metrics['loss'] = loss.item()
        for metric in batch_metrics:
            step_metrics[metric].append(batch_metrics[metric])

    for metric in step_metrics:
        step_metrics[metric] = [np.mean(step_metrics[metric])]

    return pd.DataFrame.from_dict(step_metrics)

def val_step(model, val_dataloader, criterion, device, averaging_method='micro', num_classes:int|None=None):
    model.eval()

    step_metrics = {'acc':[], 'recall':[], 'precision':[], 'f1':[], 'loss':[]}

    with torch.inference_mode():
        for i, (samples, targets) in tqdm(enumerate(val_dataloader), desc='Validation Step'):
            samples, targets = samples.to(device), targets.to(device)

            output = model(samples)

            loss = criterion(output, targets)

            _, preds = torch.max(output, 1)
            batch_metrics = utils.calc_metrics(preds, targets, average=averaging_method, num_classes=num_classes)
            batch_metrics['loss'] = loss.item()
            for metric in batch_metrics:
                step_metrics[metric].append(batch_metrics[metric])

    for metric in step_metrics:
        step_metrics[metric] = [np.mean(step_metrics[metric])]

    return pd.DataFrame.from_dict(step_metrics)

def test(model, test_dataloader, device, averaging_method='micro', num_classes:int|None=None):
    model.eval()

    metrics = {'acc':[], 'recall':[], 'precision':[], 'f1':[]}

    with torch.inference_mode():
        for i, (samples, targets) in enumerate(test_dataloader):
            samples, targets = samples.to(device), targets.to(device)

            output = model(samples)
            
            _, preds = torch.max(output, 1)
            batch_metrics = utils.calc_metrics(preds, targets, average=averaging_method, num_classes=num_classes)
            for metric in batch_metrics:
                metrics[metric].append(batch_metrics[metric])

    for metric in metrics:
        metrics[metric] = [np.mean(metrics[metric])]

    return pd.DataFrame.from_dict(metrics)

def train_loop(epochs:int, model, train_dataloader, val_dataloader, criterion, optimizer, device, lr_scheduler=None, model_ema=None, model_ema_steps=1, early_stopper:utils.EarlyStopper|None=None, averaging_method='micro', num_classes=None, output_dir=''):
    epochs_train_metrics = pd.DataFrame.from_dict({'acc':[], 'recall':[], 'precision':[], 'f1':[], 'loss':[], 'lr':[],})
    epochs_val_metrics = pd.DataFrame.from_dict({'acc':[], 'recall':[], 'precision':[], 'f1':[], 'loss':[],})

    checkpoints_dir_path = 'checkpoints'
    if output_dir:
        checkpoints_dir_path = os.path.join(output_dir, checkpoints_dir_path)
    os.makedirs(checkpoints_dir_path, exist_ok=True)

    for epoch in range(epochs):
        print('-'*50)
        print(f'Epoch: {epoch+1}/{epochs}')

        # perform a train step
        start_time = time.time()
        train_metrics = train_step(
            model, 
            train_dataloader, 
            optimizer, 
            criterion, 
            device,
            model_ema=model_ema,
            model_ema_steps=model_ema_steps,
            averaging_method=averaging_method,
            num_classes=num_classes,
            )
        print(f'Total step time: {time.time()-start_time} seconds')

        # adjust learning rate
        curr_lr = None
        if lr_scheduler is not None:
            lr_scheduler.step()
            curr_lr = lr_scheduler.get_last_lr()
        else:
            curr_lr = optimizer.param_groups[0]['lr']
        train_metrics['lr'] = curr_lr
        
        print(f'Training metrics: \n{train_metrics.to_string(index=False)}')
        print('+'*50)

        # perform a val step
        start_time = time.time()
        val_metrics = val_step(
            model, 
            val_dataloader, 
            criterion, 
            device,
            averaging_method=averaging_method,
            num_classes=num_classes,
            )
        print(f'Total step time: {time.time()-start_time} seconds')
        print(f'Validation metrics: \n{val_metrics.to_string(index=False)}')
        
        # aggregate metrics
        epochs_train_metrics = pd.concat([epochs_train_metrics, train_metrics])
        epochs_val_metrics = pd.concat([epochs_val_metrics, val_metrics])
        
        # perform early stopping
        if early_stopper is not None:
            if early_stopper.early_stop(val_metrics['loss'].item()):
                cprint(colored(f'Stopping early at epoch {epoch+1} of {epochs} because validation loss did not improve by at least {early_stopper.min_delta} for {early_stopper.patience} epochs', color='red'))
                break
            else:
                if early_stopper.counter != 0:
                    # validation loss must not have improved significantly
                    # don't save checkpoints now, 
                    # as they'd technically be worse than those before
                    cprint(colored(f'Not saving checkpoint for epoch {epoch+1} as validation loss did not decrease by at least {early_stopper.min_delta}', color='yellow'))
                    cprint(colored(f'Stopping early in {early_stopper.patience-early_stopper.counter} epoch(s) if validation loss does not improve by at least {early_stopper.min_delta}', color='yellow'))
                    continue

        # didn't stop early and/or validation loss improved by at least min_delta, so save checkpoint
        cprint(colored(f'Saving checkpoint for epoch {epoch+1}!', color='green'))
        utils.save_checkpoint(
            model, 
            optimizer, 
            epoch+1, 
            lr_scheduler=lr_scheduler, 
            output_dir=checkpoints_dir_path,
            )
        
    return epochs_train_metrics, epochs_val_metrics
        
def main(args:argparse.Namespace):
    # save command line args to disk
    cmd_args_output_path = 'cmd_args.json'
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        cmd_args_output_path = os.path.join(args.output_dir, cmd_args_output_path)
    with open(cmd_args_output_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    print('Making dataset transforms...')
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
    
    print('Making dataloaders...')
    train_dataloader = utils.make_dataloader(
        train_dataset, 
        gen_default_dataloader_config() if args.train_dataloader_config is None else json.load(open(args.train_dataloader_config)),
        )
    val_dataloader = utils.make_dataloader(
        val_dataset, 
        gen_default_dataloader_config() if args.val_dataloader_config is None else json.load(open(args.val_dataloader_config)),
        )
    test_dataloader = utils.make_dataloader(
        test_dataset, 
        gen_default_dataloader_config() if args.test_dataloader_config is None else json.load(open(args.test_dataloader_config)),
        )
    print(f'Samples in train dataset: {len(train_dataloader.dataset)}')
    print(f'Samples in val dataset: {len(val_dataloader.dataset)}')
    print(f'Samples in test dataset: {len(test_dataloader.dataset)}')

    print(f'Creating {args.model} model...')
    device = torch.device(args.device)
    model_config = {}
    if args.model_config is not None:
        model_config = json.load(open(args.model_config))
    model_config['num_classes'] = num_classes
    model = torchvision.models.get_model(args.model, **model_config)
    model_ema, model_ema_config = None, None
    if args.model_ema_config is not None:
        print('Creating model EMA...')
        model_ema_config = json.load(open(args.model_ema_config))
        model_ema = utils.make_model_ema(model, model_ema_config=model_ema_config)
        model_ema.to(device)
    model.to(device)

    print(f'Creating {args.optimizer} optimizer...')
    optimizer_config = {}
    if args.optimizer_config is not None:
        optimizer_config = json.load(open(args.optimizer_config))
    optimizer_config['lr'] = args.lr
    optimizer = utils.make_optimizer(
        model.parameters(), 
        args.optimizer, 
        optimizer_config=optimizer_config,
        )

    print('Creating loss function...')
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = None
    if args.lr_scheduler is not None:
        print(f'Creating {args.lr_scheduler} learning rate scheduler...')
        if args.lr_scheduler_config is None:
            raise Exception('Must supply a learning rate scheduler config if a learning rate scheduler is specified')
        lr_scheduler = utils.make_lr_scheduler(
            optimizer, 
            args.lr_scheduler, 
            json.load(open(args.lr_scheduler_config)),
            )

    early_stopper = None
    if args.early_stopper_config is not None:
        print('Creating early stopper...')
        early_stopper = utils.make_early_stopper(json.load(open(args.early_stopper_config)))

    print('Training...')
    train_metrics, val_metrics = train_loop(
        args.epochs, 
        model, 
        train_dataloader, 
        val_dataloader, 
        criterion, 
        optimizer, 
        device, 
        lr_scheduler=lr_scheduler,
        model_ema=model_ema,
        model_ema_steps=model_ema_config['steps'] if model_ema_config is not None else 1,
        early_stopper=early_stopper,
        averaging_method=args.averaging_method,
        num_classes=num_classes,
        output_dir=args.output_dir,
        )
    
    print('Testing...')
    test_metrics = test(
        model, 
        test_dataloader, 
        device, 
        averaging_method=args.averaging_method, 
        num_classes=num_classes,
        )

    print('Saving metrics...')
    metrics_dir_path = 'metrics'
    if args.output_dir is not None:
        metrics_dir_path = os.path.join(args.output_dir, metrics_dir_path)
    os.makedirs(metrics_dir_path, exist_ok=True)
    curr_time = int(time.time())
    train_metrics.to_csv(os.path.join(metrics_dir_path, f'train_metrics_{curr_time}.csv'), index=False)
    val_metrics.to_csv(os.path.join(metrics_dir_path, f'val_metrics_{curr_time}.csv'), index=False)
    test_metrics.to_csv(os.path.join(metrics_dir_path, f'test_metrics_{curr_time}.csv'), index=False)
    print(f'Test metrics: \n{test_metrics.to_string(index=False)}')

    if args.plot_metrics:
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

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
