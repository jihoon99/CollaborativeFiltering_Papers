import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
from datetime import datetime
# from utils import load_config, set_logger, print_to_json, print_to_list
# from utils.torch_utils import seed_everything
# import src
import gc
import argparse
import logging
import os
from pathlib import Path

from utils.core_utils import load_yaml

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', default='./configs/SimpleX_AmazonBooks/model_config.yaml')
    p.add_argument('--data_config', default='./configs/SimpleX_AmazonBooks/dataset_config.yaml')
    args = vars(p.parse_args())
    return args


if __name__ == "__main__":
    config = define_argparser()

    model_config = load_yaml(config['model_config'])
    data_config = load_yaml(config['data_config'])
    config.update(model_config)

    print(config)

    # print(type(config))
    # config.update(model_config)


