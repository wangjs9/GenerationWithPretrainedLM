# coding=utf-8

import json
import os
import logging
import torch
from os.path import join

from BlenderbotModel import BlenderbotSmall
from transformers import (AutoTokenizer, AutoModel, AutoConfig)
from torch.distributed import get_rank

logger = logging.getLogger(__name__)


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def build_model(only_toker=False, checkpoint=None, local_rank=-1, config=None):
    if 'model_name' not in config or 'pretrained_model_path' not in config:
        raise ValueError
    toker = AutoTokenizer.from_pretrained(config['pretrained_model_path'])

    if only_toker:
        if 'expanded_vocab' in config:
            toker.add_tokens(config['expanded_vocab'], special_tokens=True)
        return toker

    model = BlenderbotSmall.from_pretrained(config['pretrained_model_path'])
    if config.get('custom_config_path', None) is not None:
        model = BlenderbotSmall(AutoConfig.from_pretrained(config['custom_config_path']))

    if 'gradient_checkpointing' in config:
        setattr(model.config, 'gradient_checkpointing', config['gradient_checkpointing'])

    if 'expanded_vocab' in config:
        toker.add_tokens(config['expanded_vocab'], special_tokens=True)
    model.tie_tokenizer(toker)

    if checkpoint is not None:
        if local_rank == -1 or get_rank() == 0:
            logger.info('loading finetuned model from %s' % checkpoint)
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

    return toker, model


def deploy_model(model, args, local_rank=-1):
    if local_rank == -1 or get_rank() == 0:
        logger.info('deploying model...')
    n_gpu = args.n_gpu
    device = args.device
    model.to(device)

    # if args.local_rank != -1:
    #    model = torch.nn.parallel.DistributedDataParallel(
    #        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    #    ).to(args.device)
    # el
    if n_gpu > 1:
        if local_rank == -1 or get_rank() == 0:
            logging.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)

    return model
