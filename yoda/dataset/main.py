import argparse
import os
import pickle
import yaml

import numpy as np
import tensorflow as tf

from utils import *

import tensorflow_datasets as tfds

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

physical_devices = tf.config.list_physical_devices('GPU')

os.environ['TF_DETERMINISTIC_OPS'] = '0'

for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)

parser = argparse.ArgumentParser()
parser.add_argument('--params', dest='params')
args = parser.parse_args()

with open(f'./{args.params}', 'r') as f:
    params_loaded = yaml.safe_load(f)


DATASET = params_loaded['dataset']

datadir = ['dataset', 'dataset/' + DATASET]

mkdir(datadir)

if os.path.exists(f'./dataset/{DATASET}/train') and os.path.exists(f'./dataset/{DATASET}/test') :

    train = pickle.load(open(f'./dataset/{DATASET}/train','rb'))
    test = pickle.load(open(f'./dataset/{DATASET}/test','rb'))
else:

    train, test = eval(DATASET)()

    pickle.dump(train, open(f'./dataset/{DATASET}/train','wb'))
    pickle.dump(test, open(f'./dataset/{DATASET}/test','wb'))

