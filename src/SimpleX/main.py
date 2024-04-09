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
import pandas as pd
from utils.core_utils import load_yaml
from utils.data_utils import (
    build_dataset,
    split_train_test
)

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_config', default='./configs/SimpleX_AmazonBooks/model_config.yaml')
    p.add_argument('--data_config', default='./configs/SimpleX_AmazonBooks/dataset_config.yaml')
    p.add_argument('--train_dataset_fn', default='./data/AmazonBooks/train.csv')
    
    args = vars(p.parse_args())
    return args

def load_dataset(path, valid_ratio=0.2):
    train_df = pd.read_csv(path)
    if valid_ratio:
        train_df, valid_df = split_train_test(train_df, valid_size=valid_ratio)



if __name__ == "__main__":
    config = define_argparser()

    model_config = load_yaml(config['model_config'])
    data_config = load_yaml(config['data_config'])
    config.update(model_config)

    print(config)

    # print(type(config))
    # config.update(model_config)


