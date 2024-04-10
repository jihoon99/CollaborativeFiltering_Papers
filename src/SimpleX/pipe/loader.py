import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


class TrainDataset(Dataset):
    def __init__(self, data, user_item):
        self.data = data
        self.user_item = user_item
        self.labels = data['label']
        self.user_id = data['user_id']
        self.corpus_index = data['corpus_index']
        self.pos_item_indexes = data['corpus_index']
        self.all_item_indexes = None

    def __getitem__(self, index):
        user = self.user_id[index]
        # include neg samples
        items = self.all_item_indexes[index, :]
        label = self.labels[index]
        return user, items, label
    
    def __len__(self):
        return len(self.data)
    

def collate_fn(batch):
    users, items, labels = default_collate(batch)
    num_negs = items.shape[-1] - 1
    # reshape item data with (b*(num_neg + 1) x input_dim)
    items = items.reshape(-1)
    # add negative labels
    labels = torch.cat([labels.view(-1, 1).float(), torch.zeros((labels.size(0), num_negs))], dim=1) # [1,1,1,1,1] -> [1,0,0,0,], [1,0,0,0,0]
    return users, items, labels

def TrainLoader(DataLoader):
    def __init__(
            self, 
            data, 
            number_of_items,
            user_item, 
            config,
            sampling_replace=False,
            batch_size=32, 
            shuffle=True,
            num_workers=3, 
            num_negs=500,
        ):

        self.data = data
        self.number_of_items = number_of_items
        self.user_item = user_item
        self.num_negs = num_negs
        self.dataset = TrainDataset(data, user_item)
        super(TrainDataset, self).__init__(
            dataset=self.dataset,
            batch_size = batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )        

        self.config = config
        self.sampling_replace = sampling_replace
        self.user_item = user_item
        self.num_data = len(self.dataset)
        self.num_batch = int(np.ceil(self.num_data/batch_size))

    def __len__(self):
        return self.num_batch
    
    def __iter__(self):
        self.negative_sampling(
            self.number_of_items,
            self.num_data,
            self.num_negs,
            self.sampling_replace
        )
        iter = super(TrainLoader, self).__iter__()
        while True:
            try:
                yield next(iter)
            except StopIteration:
                return
            
    def negative_sampling(
            self, 
            number_of_items,
            num_data,
            num_negs,
            sampling_replace
            ):
        
        neg_item_indexes = self.sampling(
            num_item = number_of_items,
            num_data = num_data,
            num_negs = num_negs,
            replace = sampling_replace
        )

        self.dataset.all_item_indexes = np.hstack(
            [self.dataset.pos_item_indees.reshape(-1,1),
             neg_item_indexes]
        )

    def sampling(self, num_items, num_data, num_negs, sampling_probs=None, replace=False):
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        if sampling_probs is None:
            sampling_probs = np.ones(num_items)/num_items
        sampled_array = np.random.choice(num_items,
                                        size = (num_data, num_negs),
                                        replace=replace)
        return sampled_array
    

if __name__ == "__main__":
    import pandas as pd
    from ..utils.data_utils import get_user2items_dict

    t_data = pd.read_csv("/Users/rainism/Desktop/CollaborativeFiltering_Papers/src/SimpleX/data/AmazonBooks/train.csv")
    user_item = get_user2items_dict(t_data)
    aa = TrainLoader(user_item, )