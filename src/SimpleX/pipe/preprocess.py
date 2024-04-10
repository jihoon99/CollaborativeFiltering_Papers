import pandas as pd
import numpy as np

def load_data(data_fn):
    if data_fn.endswith(".csv"):
        df = pd.read_csv(data_fn)
    elif data_fn.endswith(".pkl"):
        df = pd.read_pickle(data_fn)
    return df


def transform_dataset(data_fn):
    