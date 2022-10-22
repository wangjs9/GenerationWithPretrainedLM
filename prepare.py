# coding=utf-8

import argparse
import json
import shutil
import pickle
import os
import logging
import multiprocessing as mp
from os.path import dirname, exists, join
from utils.data_process import convert_data_to_inputs, convert_inputs_to_features
import torch
import tqdm
from utils.building_utils import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--train_input_file', type=str, default='./data/train.txt')
parser.add_argument('--max_input_length', type=int, default=160, help='discard data longer than this')
parser.add_argument('--max_decoder_input_length', type=int, default=40, help='discard data longer than this')
parser.add_argument('--single_processing', action='store_true', help='do not use multiprocessing')

args = parser.parse_args()

model_config = {
    "model_name": "vanilla_blenderbot_small",
    "pretrained_model_path": "./models/Blenderbot_small-90M",
    "custom_config_path": None,
    "gradient_checkpointing": None
}


toker = build_model(only_toker=True, config=model_config)

with open(args.train_input_file) as f:
    reader = f.readlines()

if not os.path.exists(f'./DATA'):
    os.mkdir(f'./DATA')
save_dir = f'./DATA'
if not exists(save_dir):
    os.mkdir(save_dir)

kwargs = {
    'max_input_length': args.max_input_length,
    'max_decoder_input_length': args.max_decoder_input_length,
}


def process_data(line):
    data = json.loads(line)
    inputs = convert_data_to_inputs(
        data=data,
        toker=toker,
        **kwargs
    )
    features = convert_inputs_to_features(
        inputs=inputs,
        toker=toker,
        **kwargs,
    )
    return features


processed_data = []
if args.single_processing:
    for features in map(process_data, tqdm.tqdm(reader, total=len(reader))):
        processed_data.extend(features)
else:
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for features in pool.imap(process_data, tqdm.tqdm(reader, total=len(reader))):
            processed_data.extend(features)

# save data
data_path = f'{save_dir}/data.pkl'
with open(data_path, 'wb') as file:
    pickle.dump(processed_data, file)
kwargs.update({'n_examples': len(processed_data)})
# save relevant information to reproduce
with open(f'{save_dir}/meta.json', 'w') as writer:
    json.dump(kwargs, writer, indent=4)
torch.save(toker, f'{save_dir}/tokenizer.pt')
