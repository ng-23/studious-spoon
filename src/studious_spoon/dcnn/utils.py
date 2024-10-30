import os
import torch
import torch.utils
import torchvision
import numpy as np
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from schemas import validate_config
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torcheval.metrics.functional import multiclass_accuracy, multiclass_recall, multiclass_f1_score, multiclass_precision 

SUPPORTED_OPTIMIZERS = {
    'adam':optim.Adam, 
    'adamw':optim.AdamW, 
    'sgd':optim.SGD, 
    'rmsprop':optim.RMSprop,
    }

SUPPORTED_LR_SCHEDULERS = {
    'step_lr':optim.lr_scheduler.StepLR, 
    'exponential_lr':optim.lr_scheduler.ExponentialLR, 
    'cosineannealing_lr':optim.lr_scheduler.CosineAnnealingLR,
    }

class EarlyStopper:
    '''
    See https://stackoverflow.com/a/76602544
    '''

    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience  # number of times to allow for no improvement before stopping the execution
        self.min_delta = min_delta  # the minimum change to be counted as improvement
        self.counter = 0  # count the number of times the validation accuracy not improving
        self.min_validation_loss = np.inf

    # return True when validation loss is not decreased by the `min_delta` for `patience` times 
    def early_stop(self, validation_loss):
        if ((validation_loss+self.min_delta) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0  # reset the counter if validation loss decreased at least by min_delta
        
        elif ((validation_loss+self.min_delta) >= self.min_validation_loss):
            self.counter += 1 # increase the counter if validation loss is not decreased by the min_delta
            if self.counter >= self.patience:
                return True
        
        return False
    
def encode_str_list(str_list):
    return [s.encode('utf-8') for s in str_list]

def make_transforms_pipeline(transforms_config:dict):
    pipeline = []
        
    for transform_name in transforms_config:
        if not hasattr(transforms, transform_name):
            raise Exception(f'No such transform {transform_name}')
        
        transform_config = validate_config(transforms_config[transform_name], transform_name)

        transform_func = getattr(transforms, transform_name)
        pipeline.append(transform_func(**transform_config))
            
    return transforms.Compose(pipeline)

def load_dataset(filepath:str):
    return torchvision.datasets.ImageFolder(root=filepath)

def split_dataset(dataset:torchvision.datasets.ImageFolder, train_ratio:float=0.6, val_ratio:float=0.2, test_ratio:float=0.2, seed=42):
    if train_ratio + val_ratio + test_ratio > 1.0:
        raise Exception('Train/val/test ratios must be <= 1.0')
    
    generator = torch.Generator().manual_seed(seed)
    
    return torch.utils.data.random_split(dataset, [train_ratio, val_ratio, test_ratio], generator=generator)

def make_dataloader(dataset, dataloader_config:dict|None=None):
    if dataloader_config is None:
        dataloader_config = {} # use dataloader's defaults

    dataloader_config = validate_config(dataloader_config, 'DataLoader')

    return torch.utils.data.DataLoader(dataset, **dataloader_config)

def make_optimizer(model_params, optimizer:str, optimizer_config:dict|None=None):
    optimizer = optimizer.lower()

    if optimizer_config is None:
        optimizer_config = {} # use optimizer's defaults

    if optimizer not in SUPPORTED_OPTIMIZERS:
        raise Exception(f'Unknown/unsupported optimizer {optimizer}. Supported ones are currently {list(SUPPORTED_OPTIMIZERS.keys())}')
    optim_class = SUPPORTED_OPTIMIZERS[optimizer]

    optimizer_config = validate_config(optimizer_config, optim_class.__name__)

    return optim_class(model_params, **optimizer_config)

def make_lr_scheduler(optimizer, lr_scheduler:str, lr_scheduler_config:dict):
    lr_scheduler = lr_scheduler.lower()

    if lr_scheduler not in SUPPORTED_LR_SCHEDULERS:
        raise Exception(f'Unknown/unsupported learning rate scheduler {lr_scheduler}. Supported learning rate schedulers are {list(SUPPORTED_LR_SCHEDULERS.keys())}')
    lr_scheduler_class = SUPPORTED_LR_SCHEDULERS[lr_scheduler]

    lr_scheduler_config = validate_config(lr_scheduler_config, lr_scheduler_class.__name__)

    return lr_scheduler_class(optimizer, **lr_scheduler_config)

def make_model_ema(model, model_ema_config:dict):
    model_ema_config = validate_config(model_ema_config, 'ModelEMA')

    return AveragedModel(
        model, 
        multi_avg_fn=get_ema_multi_avg_fn(decay=model_ema_config['decay']),
        )

def make_early_stopper(early_stopper_config:dict|None=None):
    if early_stopper_config is None:
        early_stopper_config = {}

    early_stopper_config = validate_config(early_stopper_config, 'EarlyStopper')

    return EarlyStopper(**early_stopper_config)

def calc_metrics(preds, targets, average='micro', num_classes:int|None=None, ):
    metrics = {}

    acc = multiclass_accuracy(preds, targets, average=average, num_classes=num_classes).item()
    recall = multiclass_recall(preds, targets, average=average, num_classes=num_classes).item()
    precision = multiclass_precision(preds, targets, average=average, num_classes=num_classes).item()
    f1 = multiclass_f1_score(preds, targets, average=average, num_classes=num_classes).item()

    metrics['acc'] = acc
    metrics['recall'] = recall
    metrics['precision'] = precision
    metrics['f1'] = f1

    return metrics

def plot_comparison(x, y1, y2, label1:str='A', label2:str='B', title:str='', x_label:str='', y_label:str='', filename='plot.png', output_dir=''):
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)

    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.legend()

    output_path = filename
    if output_dir:
        output_path = os.path.join(output_dir, output_path)
    plt.savefig(output_path)
    plt.close()

def save_checkpoint(model, optimizer, epoch, model_ema=None, lr_scheduler=None, output_dir=''):
    checkpoint = {
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'epoch':epoch,
    }
    
    if lr_scheduler is not None:
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()

    if model_ema is not None:
        checkpoint['model_ema'] = model_ema.state_dict()

    torch.save(
        checkpoint, 
        os.path.join(output_dir, f'checkpoint_{epoch}.pth'),
        )
