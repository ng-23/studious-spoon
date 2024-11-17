import argparse
import os
import json
import torchvision
import torch
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from studious_spoon.dcnn import utils
from torchvision.models.feature_extraction import create_feature_extractor

def get_args_parser():
    parser = argparse.ArgumentParser(
        description='-= PyTorch DCNN feature extraction script -=', 
        add_help=True,
        )
    
    parser.add_argument(
        'dataset_path',
        metavar='dataset-path',
        type=str,
        help='Filepath to image dataset, structured like ImageNet',
        )
    
    parser.add_argument(
        'model', 
        type=str, 
        choices=list(FeatureExtractor.supported_models), 
        help='DCNN model to load weights into and extract features from',
        )
    
    parser.add_argument(
        'checkpoint_path', 
        metavar='checkpoint-path', 
        type=str, 
        help='Filepath to model checkpoint to load weights from',
        )
    
    parser.add_argument(
        'layer', 
        type=str, 
        help='Layer of model to extract features from',
        )
    
    parser.add_argument(
        '--transforms-config', 
        type=str, 
        default=None, 
        help='Path to JSON configuration file for dataset transforms',
        )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=64, 
        help='Dataloader batch size',
        )
    
    parser.add_argument(
        '--num-workers', 
        type=int, 
        default=0, 
        help='Number of worker processes to spawn when loading data',
        )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda', 
        help='Hardware device to use',
        )
    
    parser.add_argument(
        '--flatten-features', 
        action='store_true', 
        help='If specified, flatten feature vectors before saving',
        )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Seed to use for controlling randomness',
        )
    
    parser.add_argument(
        '--dataset-name', 
        type=str, 
        choices=['train','val','test'], 
        default='train', 
        help='Name of dataset when saving',
        )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None, 
        help='Filepath of directory to save features dataset to',
        )
    
    return parser

class FeatureExtractor:
    '''
    Feature extractor for DCNNs

    DCNN architectures:
    - AlexNet, VGG, GoogLeNet - https://www.digitalocean.com/community/tutorials/popular-deep-learning-architectures-alexnet-vgg-googlenet
    - ResNet - https://arxiv.org/pdf/1512.03385, https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
    - GoogLeNet/InceptionV1 - https://arxiv.org/pdf/1409.4842, https://viso.ai/deep-learning/googlenet-explained-the-inception-model-that-won-imagenet/
    - EfficientNet - https://towardsdatascience.com/complete-architectural-details-of-all-efficientnet-models-5fd5b736142
    '''

    supported_models = {'alexnet','vgg16','vgg19','resnet18','googlenet',}

    alexnet_nodes = {
        'conv1':'features.0', 
        'conv2':'features.3', 
        'conv3':'features.6',
        'conv4':'features.8',
        'conv5':'features.10',

        'fc7':'classifier.1',
        'fc8':'classifier.4'
        }
    
    vgg16_nodes = {
        'conv1-1':'features.0',
        'conv1-2':'features.2',

        'conv2-1':'features.5',
        'conv2-1':'features.7',

        'conv3-1':'features.10',
        'conv3-2':'features.12',
        'conv3-3':'features.14',

        'conv4-1':'features.17',
        'conv4-2':'features.19',
        'conv4-3':'features.21',

        'conv5-1':'features.24',
        'conv5-2':'features.26',
        'conv5-3':'features.28',

        'fc6':'classifier.0',
        'fc7':'classifier.3',
        }
    
    vgg19_nodes = {
        'conv1-1':'features.0',
        'conv1-2':'features.2',

        'conv2-1':'features.5',
        'conv2-2':'features.7',

        'conv3-1':'features.10',
        'conv3-2':'features.12',
        'conv3-3':'features.14',
        'conv3-4':'features.16',
        
        'conv4-1':'features.19',
        'conv4-2':'features.21',
        'conv4-3':'features.23',
        'conv4-5':'features.25',
        
        'conv5-1':'features.28',
        'conv5-2':'features.30',
        'conv5-3':'features.32',
        'conv5-4':'features.34',

        'fc6':'classifier.0',
        'fc7':'classifier.3',
        }
    
    resnet18_nodes = {
        'conv2-1':'layer1.0.conv1',
        'conv2-2':'layer1.0.conv2',
        'conv2-3':'layer1.1.conv1',
        'conv2-4':'layer1.1.conv2',

        'conv3-1':'layer2.0.conv1',
        'conv3-2':'layer2.0.conv2',
        'conv3-3':'layer2.1.conv1',
        'conv3-4':'layer2.1.conv2',

        'conv4-1':'layer3.0.conv1',
        'conv4-2':'layer3.0.conv2',
        'conv4-3':'layer3.1.conv1',
        'conv4-4':'layer3.1.conv2',

        'conv5-1':'layer4.0.conv1',
        'conv5-2':'layer4.0.conv2',
        'conv5-3':'layer4.1.conv1',
        'conv5-4':'layer4.1.conv2',
        }
    
    googlenet_nodes = {
        'conv1':'conv1.conv',
        'conv2':'conv2.conv',
        'conv3':'conv3.conv',
        'inception3a':'inception3a.cat',
        'inception3b':'inception3b.cat',
        'inception4a':'inception4a.cat',
        'inception4b':'inception4b.cat',
        'inception4c':'inception4c.cat',
        'inception4d':'inception4d.cat',
        'inception4e':'inception4e.cat',
        'inception5a':'inception5a.cat',
        'inception5b':'inception5b.cat',
        }
    
    def __init__(self, model, model_name, layer):
        self.model = model
        self.model_name = model_name
        self.layer = layer

    def _get_nodes_dict(self, model_name:str) -> dict:
        if not hasattr(FeatureExtractor, f'{model_name}_nodes'):
            raise Exception(f'No such nodes dict for model {model_name}')
        
        return getattr(FeatureExtractor, f'{model_name}_nodes')

    def extract_features(self, dataloader, device, flatten=False):
        model_nodes = self._get_nodes_dict(self.model_name)

        if self.layer not in model_nodes:
            raise Exception(f'{self.model_name} has no node(s) for layer {self.layer}. Valid layers are {list(model_nodes.keys())}')
        
        node = model_nodes[self.layer]
        
        fex_model = create_feature_extractor(self.model, return_nodes=[node])

        # maps an encoded class name to its string name
        # ex: {0:'car', 1:'boat'}
        encoded_labels_mapping = {v: k for k, v in dataloader.dataset.class_to_idx.items()}

        # features[i] is the feature vector for image class instance labels[i]
        img_features, img_labels = [], []

        with torch.inference_mode():
            for i, (samples, targets) in enumerate(tqdm(dataloader, desc='Processing batches...')):
                samples, targets = samples.to(device), targets.to(device)

                # output is a dict mapping a node (eg fc)
                # to a tensor of tensors representing the features for each image in the batch
                output = fex_model(samples)

                features = output[node]

                for j in range(len(features)):
                    img_features.append(features[j].cpu().numpy())
                    img_labels.append(encoded_labels_mapping[targets[j].cpu().item()])

                    if flatten:
                        img_features[-1] = img_features[-1].flatten()

        return np.array(img_features), np.array(img_labels)

def save_as_dataset(features, labels, dataset_name, filename='', output_dir=''):
    if not filename:
        filename = 'dataset_cache.h5'
    
    output_path = filename
    if output_dir:
        output_path = os.path.join(output_dir, filename)

    with h5py.File(output_path, 'w') as h5_file:
        # save image features
        h5_file.create_dataset(f'imgs/{dataset_name}', data=features)
        
        # save labels as strings in HDF5 (strings need to be encoded to bytes)
        h5_file.create_dataset(f'labels/{dataset_name}', data=utils.encode_str_list(labels))
    
def main(args:argparse.Namespace):
    # save command line args to disk
    cmd_args_output_path = 'cmd_args.json'
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        cmd_args_output_path = os.path.join(args.output_dir, cmd_args_output_path)
    with open(cmd_args_output_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    print('Making dataset transforms...')
    dataset_transforms = utils.make_dataset_transforms(None if args.transforms_config is None else json.load(open(args.transforms_config)))

    print('Loading dataset...')
    dataset = utils.load_dataset(args.dataset_path)
    dataset.transform = dataset_transforms
    num_classes = len(dataset.classes)
    print(f'Dataset has {len(dataset)} samples of {num_classes} classes')
    
    print('Making dataloaders...')
    dl_config = utils.gen_standard_dataloader_config()
    dl_config['batch_size'] = args.batch_size; dl_config['num_workers'] = args.num_workers
    dataloader = utils.make_dataloader(dataset, dataloader_config=dl_config)

    print(f'Creating {args.model} model...')
    device = torch.device(args.device)
    model = torchvision.models.get_model(args.model, num_classes=num_classes)
    model_checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(model_checkpoint['model'])
    model.to(device)

    print('Extracting features...')
    fex = FeatureExtractor(model, args.model, args.layer)
    features, labels = fex.extract_features(dataloader, device, flatten=args.flatten_features)

    print('Randomly shuffling features and labels...')
    features, labels = shuffle(features, labels, random_state=args.seed)

    print('Saving features...')
    save_as_dataset(features, labels, args.dataset_name, output_dir=args.output_dir)

    print('Done!')

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)