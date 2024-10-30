import os
import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from feature_extraction import save_as_dataset

'''
Script for performing fusion of features extracted from the same images
'''

def get_args_parser():
    parser = argparse.ArgumentParser(
        description='-= Feature fusion script -=', 
        add_help=True,
        )
    
    parser.add_argument(
        'features_paths', 
        metavar='features-paths', 
        nargs='+', 
        help='Filepaths to each of the feature datasets to fuse',
        )
    
    parser.add_argument(
        '--load-dataset-names', 
        type=str, 
        choices=['train','val','test'], 
        default=None, 
        nargs='+',
        help='Dataset name for each feature dataset to fuse features from when loading. If unspecified, "train" is assumed to be the dataset name for each provided features dataset',
        )
    
    parser.add_argument(
        '--fusion-method', 
        type=str, 
        choices=list(FeatureFuser.supported_fusion_methods), 
        default='max', 
        help='Feature fusion method to use. Other than concat, all fusion methods are performed element-wise',
        )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Seed to use for controlling randomness',
        )
    
    parser.add_argument(
        '--save-dataset-name', 
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

class FeaturesDataset():
    def __init__(self, dataset_paths:list[str], dataset_names:list[str]):
        self.dataset_paths = dataset_paths
        self.dataset_names = dataset_names

        if len(self.dataset_paths) == 0 or len(self.dataset_names) == 0:
            raise Exception('Must have at least 1 dataset path and name')
        
        if len(self.dataset_paths) != len(self.dataset_names):
            raise Exception(f'The number of dataset names ({len(self.dataset_names)}) does not match the number of features datasets ({len(self.dataset_paths)}) when loading')
        
        # maps an image name (eg car) to a list of feature vectors (eg [[1,2,3],[4,5,6]]) from each features dataset loaded
        self._dataset = self._load_datasets_from_paths()

    def _load_datasets_from_paths(self):
        # maps an image label to an array of that image's feature vectors from each features dataset
        # ex: {'car':{0:[...,...,...],1:[...,...,...],2:[...,...,...]}}
        res = {}

        for i in tqdm(range(len(self.dataset_paths)), desc='Loading feature vectors...'):
            path, name = self.dataset_paths[i], self.dataset_names[i]

            label_counts = {}

            # we assume there is 1:1 ratio between labels and image feature vectors
            with h5py.File(path, mode='r') as f:
                labels, imgs_data = f[f'labels/{name}'], f[f'imgs/{name}']

                for j in range(len(labels)):
                    label, img_data = labels[j].decode(), imgs_data[j]

                    if label in res:
                        if label in label_counts:
                            label_counts[label] += 1
                        else:
                            label_counts[label] = 1

                        label_instance = label_counts[label]

                        if label_instance in res[label]:
                            res[label][label_instance].append(img_data)
                        else:
                            res[label][label_instance] = [img_data]
                    else:
                        res[label] = {1:[img_data]}
                        label_counts[label] = 1

        return res

    def get_img_labels(self):
        return list(self._dataset.keys())
    
    def get_img_features(self, img_name:str):
        if img_name not in self._dataset:
            raise Exception(f'No such image {img_name} in features dataset')
        
        return self._dataset[img_name]

class FeatureFuser():
    supported_fusion_methods = {'max', 'min', 'avg', 'mul', 'sum', 'concat',}
    fusion_method_funcs = {'max':np.maximum.reduce, 'min':np.minimum.reduce, 'avg':np.mean, 'mul':np.multiply.reduce, 'sum':np.sum, 'concat':np.concat,}

    def __init__(self, features_dataset:FeaturesDataset, fusion_method:str='max'):
        self.features_dataset = features_dataset
        self.fusion_method = fusion_method

    def _validate_fusion_method(self):
        if self.fusion_method not in FeatureFuser.supported_fusion_methods:
            raise Exception(f'Invalid fusion method {self.fusion_method} - valid fusion methods are currently {list(FeatureFuser.supported_fusion_methods)}')

    def fuse_features(self):
        '''
        Fuse feature vectors for each image using specified fusion method
        '''

        self._validate_fusion_method()
        fusion_func = FeatureFuser.fusion_method_funcs[self.fusion_method]

        features, labels = [], []

        for label in tqdm(self.features_dataset.get_img_labels(), desc='Fusing features...'):
            feature_vecs = self.features_dataset.get_img_features(label)
            for i in range(len(feature_vecs)):
                labels.append(label)
                features.append(fusion_func(feature_vecs[i+1], axis=0))

        return features, labels

def main(args):
    # ensure 1:1 ratio of feature datasets and dataset names for loading
    if args.load_dataset_names is None:
        args.load_dataset_names = ['train' for _ in range(len(args.features_paths))]
        
    # save validated command line args to disk
    cmd_args_output_path = 'cmd_args.json'
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        cmd_args_output_path = os.path.join(args.output_dir, cmd_args_output_path)
    with open(cmd_args_output_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # load feature datasets
    print('Loading feature datasets...')
    features_dataset = FeaturesDataset(
        args.features_paths, 
        args.load_dataset_names, 
        )

    # fuse features for each image
    print(f'Fusing features using {args.fusion_method} method...')
    fef = FeatureFuser(features_dataset, fusion_method=args.fusion_method,)
    features, labels = fef.fuse_features()

    print('Randomly shuffling features and labels...')
    features, labels = shuffle(features, labels, random_state=args.seed)

    # save fused features as an hdf5 file to disk
    print('Saving features...')
    save_as_dataset(
        features, 
        labels, 
        args.save_dataset_name, 
        filename='fused_dataset_cache.h5', 
        output_dir=args.output_dir,
        )

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)