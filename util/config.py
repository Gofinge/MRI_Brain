import argparse

import os
import torch
import yaml


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D training')
    parser.add_argument('--config', default='resources/train_config.yaml', type=str, help='Path to the YAML config file')
    args = parser.parse_args()
    config = _load_config_yaml(args.config)
    # Get a device to train on
    DEFAULT_DEVICE = 'cuda:0'
    device = config.get('device', DEFAULT_DEVICE)
    config['device'] = torch.device(device)
    return config


def _load_config_yaml(config_file):
    return yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
