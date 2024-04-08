import os
import sys
import yaml
import glob
import json


def load_yaml(yaml_dir):
    with open(yaml_dir, 'r') as cfg:
        config = yaml.load(cfg, Loader=yaml.Loader)
    return config

