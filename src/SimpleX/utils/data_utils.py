import h5py
import os
import logging
import numpy as np
import multiprocessing as mp
import gc


def split_train_test(train_df=None,  valid_ratio=0):
    train_len = len(train_df)
    train_idx = np.arange(train_len)
    np.random.shuffle(train_idx)
    if valid_ratio > 0:
        valid_len = int(train_len * valid_ratio)
        train_len = train_len - valid_len
        valid_df = train_df.loc[train_idx[train_len:], :].reset_index()
        train_df = train_df.loc[train_idx[0:train_len], :].reset_index()
    return train_df, valid_df


# def build_dataset(feature_encoder, item_corpus=None, train_data=None, valid_data=None, 
#                   test_data=None, valid_size=0, test_size=0, split_type="sequential", **kwargs):
#     """ Build feature_map and transform h5 data """
    
#     # Load csv data
#     train_ddf = feature_encoder.read_csv(train_data, **kwargs)         # ./data/Amazon/AmazonBooks_m1/train.csv     -> query_index, corpus_index, label, user_id, user_history
#     valid_ddf = None
#     test_ddf = None

#     # Split data for train/validation/test
#     if valid_size > 0 or test_size > 0:
#         valid_ddf = feature_encoder.read_csv(valid_data, **kwargs)
#         test_ddf = feature_encoder.read_csv(test_data, **kwargs)
#         train_ddf, valid_ddf, test_ddf = split_train_test(train_ddf, valid_ddf, test_ddf, 
#                                                           valid_size, test_size, split_type)

#     # fit feature_encoder
#     corpus_ddf = feature_encoder.read_csv(item_corpus, **kwargs)      # item_corpus : ./data/amazon/amazonbooks_m1/item_corpus.csv
#     corpus_ddf = feature_encoder.preprocess(corpus_ddf)               # corpus_ddf : fill na, active columns
#     train_ddf = feature_encoder.preprocess(train_ddf)
#     feature_encoder.fit(train_ddf, corpus_ddf, **kwargs)

#     # transform corpus_ddf
#     item_corpus_dict = feature_encoder.transform(corpus_ddf)
#     save_h5(item_corpus_dict, os.path.join(feature_encoder.data_dir, 'item_corpus.h5'))
#     del item_corpus_dict, corpus_ddf
#     gc.collect()

#     # transform train_ddf
#     block_size = int(kwargs.get("data_block_size", 0)) # Num of samples in a data block
#     transform_h5(feature_encoder, train_ddf, 'train.h5', preprocess=False, block_size=block_size)
#     del train_ddf
#     gc.collect()

#     # Transfrom valid_ddf
#     if valid_ddf is None and (valid_data is not None):
#         valid_ddf = feature_encoder.read_csv(valid_data, **kwargs)
#     if valid_ddf is not None:
#         transform_h5(feature_encoder, valid_ddf, 'valid.h5', preprocess=True, block_size=block_size)
#         del valid_ddf
#         gc.collect()

#     # Transfrom test_ddf
#     if test_ddf is None and (test_data is not None):
#         test_ddf = feature_encoder.read_csv(test_data, **kwargs)
#     if test_ddf is not None:
#         transform_h5(feature_encoder, test_ddf, 'test.h5', preprocess=True, block_size=block_size)
#         del test_ddf
#         gc.collect()
#     logging.info("Transform csv data to h5 done.")




if __name__ == "__main__":
    import pandas as pd
    train_df = pd.read_csv('/Users/rainism/Desktop/CollaborativeFiltering_Papers/src/SimpleX/data/AmazonBooks/train.csv')
    train_df, valid_df = split_train_test(train_df, 0.3)
    print(1)
