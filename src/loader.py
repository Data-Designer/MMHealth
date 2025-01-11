# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : loader.py
# Time       ：4/11/2024 3:25 pm
# Author     ：XXXXX
# version    ：python 
# Description：
"""
import torch
from torch.utils.data import DataLoader, Dataset
from baseline_utils import collate_fn_corgan, collate_fn_smart, collate_fn_prism



def collate_fn_dict(batch):
    return {key: [d[key] for d in batch] for key in batch[0]} # conditions: [B,V,M]

def get_dataloader(dataset, batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_dict):
    """for first, third stage"""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last
    )

    return dataloader




# def get_note():
#     pass
#
def collate_fn_note(batch):
    data = {key: [d[key] for d in batch] for key in batch[0]} #
    return data # conditions: [B,V,M]


def get_special_input(config):
    if config['MODEL']== 'PRISM':
        return collate_fn_prism
    elif config['DATASET'] == 'MIV-Note':
        return collate_fn_note
    else:
        return collate_fn_dict
