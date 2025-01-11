# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : config.py
# Time       ：4/11/2024 3:27 pm
# Author     ：XXXXXXXX
# version    ：python 
# Description：
"""

MIII_PARAMS = {
    'FEATURE' : ['conditions', 'procedures', 'drugs'],
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '4',
    'EPOCH': 50,
    'WARM_EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3,
    'BATCH': 32,
    'DROPOUT': 0.1,
    'WD': 0.,#1e-3,
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
    'AUX' : 0.00001,

}

eICU_PARAMS = {
    'FEATURE': ['conditions', 'procedures', 'drugs'],
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '2',
    'EPOCH': 50,
    'WARM_EPOCH': 50,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3, #
    'BATCH': 32,
    'DROPOUT': 0.3,
    'WD': 5e-4,
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
    'AUX': 0.001,

}

MIV_Note_PARAMS = {
    'FEATURE' : ['conditions', 'procedures', 'drugs'],
    'SEED': 528,
    'USE_CUDA': True,
    'GPU': '0',
    'EPOCH': 50,
    'WARM_EPOCH': 10,
    'DIM': 128,
    'HIDDEN': 128,
    'LR': 1e-3,
    'BATCH': 32,
    'DROPOUT': 0.3, # 0. # PHE
    'WD': 5e-4, #0.
    'RNN_LAYERS': 2,
    'MULTI': 0,
    'MAXSEQ': 10,
    'MAXCODESEQ': 512,
    'AUX': 0.001,

}


class PHECONFIG():
    """DRL config"""
    # data_info parameter
    DEV = False 
    MODEL = "ours"
    PLM = "Sap-BERT"
    TASK = 'PHE'
    DATASET = 'MIV-Note'
    LABEL = 'labels'

    ATCLEVEL = 3
    RATIO = 0.6 # train-test split
    THRES = 0.4 # pred threshold
    # train parameter

    DATASET_PARAMS = {
        'MIII': MIII_PARAMS,
        'eICU': eICU_PARAMS,
        'MIV-Note': MIV_Note_PARAMS,
    }

    @classmethod
    def get_params(cls):
        return cls.DATASET_PARAMS.get(cls.DATASET, {})

    # log
    LOGDIR = '/home/XXXX/MMHealth/log/ckpt/'



config = {**vars(PHECONFIG), **PHECONFIG.get_params()}
config = {k: v for k, v in config.items() if not k.startswith('__')}
