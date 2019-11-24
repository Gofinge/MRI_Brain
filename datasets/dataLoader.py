import collections
import os
import h5py
import numpy as np
import torch
import datasets.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from util.utils import get_logger


class HDF5Dataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, file_path, phase, transformer_config,
                 raw_internal_path='raw', label_internal_path='label',
                 weight_internal_path=None):
        """
        :param file_path: path to H5 file containing raw data as well as labels and per pixel weights (optional)
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :param transformer_config: data augmentation configuration
        :param raw_internal_path (str or list): H5 internal path to the raw dataset
        :param label_internal_path (str or list): H5 internal path to the label dataset
        :param weight_internal_path (str or list): H5 internal path to the per pixel weights
        """
        assert phase in ['train', 'val', 'test']
        # self._check_patch_shape(patch_shape)
        self.phase = phase
        self.file_path = file_path

        # convert raw_internal_path, label_internal_path and weight_internal_path to list for ease of computation
        if isinstance(raw_internal_path, str):
            raw_internal_path = [raw_internal_path]
        if isinstance(label_internal_path, str):
            label_internal_path = [label_internal_path]
        if isinstance(weight_internal_path, str):
            weight_internal_path = [weight_internal_path]

        with h5py.File(os.path.join(self.file_path, raw_internal_path), 'r') as input_file:
            # WARN: we load everything into memory due to hdf5 bug when reading H5 from multiple subprocesses, i.e.
            # File "h5py/_proxy.pyx", line 84, in h5py._proxy.H5PY_H5Dread
            # OSError: Can't read data (inflate() failed)
            self.raws = input_file['data'][...]
            # calculate global mean and std for Normalization augmentation
            mean, std = self._calculate_mean_std(self.raws[0])
            self.transformer = transforms.get_transformer(transformer_config, mean, std, phase)
            self.raw_transform = self.transformer.raw_transform()

            if phase != 'test':
                # create label/weight transform only in train/val phase
                self.labels = np.loadtxt(os.path.join(self.file_path, label_internal_path), delimiter=",")
                if weight_internal_path is not None:
                    # look for the weight map in the raw file
                    self.weight_maps = [input_file[internal_path][...] for internal_path in weight_internal_path]
                    self.weight_transform = self.transformer.weight_transform()
                else:
                    self.weight_maps = None

            else:
                # 'test' phase used only for predictions so ignore the label dataset
                self.labels = None
                self.weight_maps = None

            self.sample_count = len(self.raws)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the raw data patch for a given slice
        raw_transformed = self._transform(self.raws, idx, self.raw_transform)

        if self.phase == 'test':
            # just return the transformed raw patch and the metadata
            return raw_transformed, idx
        else:
            label = torch.FloatTensor(self.labels[idx])
            return raw_transformed, label

    @staticmethod
    def _transform(datasets, idx, transformer):
        # get the label data and apply the label transformer
        transformed_patch = transformer(datasets[idx])

        # if transformed_patches is a singleton list return the first element only
        if len(transformed_patch) == 1:
            return transformed_patch[0]
        else:
            return transformed_patch

    def __len__(self):
        return self.sample_count

    @staticmethod
    def _calculate_mean_std(input):
        """
        Compute a mean/std of the raw stack for normalization.
        This is an in-memory implementation, override this method
        with the chunk-based computation if you're working with huge H5 files.
        :return: a tuple of (mean, std) of the raw data
        """
        return input.mean(keepdims=True), input.std(keepdims=True)


def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders
    (torch.utils.data.DataLoader) backed by the datasets.hdf5.HDF5Dataset.

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger = get_logger('HDF5Dataset')
    logger.info('Creating training and validation set loaders...')

    # get train and validation files
    train_path = loaders_config['train_path']
    val_path = loaders_config['val_path']
    assert isinstance(train_path, list)
    assert isinstance(val_path, list)
    # get h5 internal paths for raw and label
    raw_internal_path = loaders_config['raw_internal_path']
    label_internal_path = loaders_config['label_internal_path']
    weight_internal_path = loaders_config.get('weight_internal_path', None)

    logger.info(f'Loading training set from: {train_path}...')
    # create H5 backed training and validation dataset with data augmentation
    train_dataset = HDF5Dataset(train_path, phase='train',
                                transformer_config=loaders_config['transformer'],
                                raw_internal_path=raw_internal_path,
                                label_internal_path=label_internal_path,
                                weight_internal_path=weight_internal_path)

    logger.info(f'Loading validation set from: {val_path}...')
    val_dataset = HDF5Dataset(val_path, phase='val',
                              transformer_config=loaders_config['transformer'],
                              raw_internal_path=raw_internal_path,
                              label_internal_path=label_internal_path,
                              weight_internal_path=weight_internal_path)

    num_workers = loaders_config.get('num_workers', 1)
    batch_size = loaders_config.get('batch_size', 1)
    logger.info(f'Number of workers for train/val datasets: {num_workers}')
    logger.info(f'Batch size for train/val datasets: {batch_size}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    }


def get_test_loaders(config):
    """
    Returns a list of DataLoader, one per each test file.

    :param config: a top level configuration object containing the 'datasets' key
    :return: generator of DataLoader objects
    """

    logger = get_logger('HDF5Dataset')

    assert 'datasets' in config, 'Could not find data sets configuration'
    loaders_config = config['loaders']

    # get train and validation files
    test_path = loaders_config['test_path']
    assert isinstance(test_path, list)
    # get h5 internal path
    raw_internal_path = loaders_config['raw_internal_path']
    # get train/validation patch size and stride

    logger.info(f'Loading test set from: {test_path}...')
    dataset = HDF5Dataset(test_path, phase='test', raw_internal_path=raw_internal_path,
                          transformer_config=loaders_config['transformer'])

    num_workers = loaders_config.get('num_workers', 1)
    batch_size = loaders_config.get('batch_size', 1)
    logger.info(f'Number of workers for test datasets: {num_workers}')
    logger.info(f'Batch size for test datasets: {batch_size}')
    return {
        'test': DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)}