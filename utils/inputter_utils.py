# coding: utf-8
import logging
import torch
from typing import List
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer


def _norm(s):
    return ' '.join(s.strip().split())


class InputFeature(object):
    def __init__(
            self,
            input_ids,
            decoder_input_ids,
            labels
    ):
        self.input_ids = input_ids
        self.input_length = len(input_ids)

        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels

        self.input_len = self.input_length + self.decoder_input_length


class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        return self.features[i]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features: List[InputFeature], tokenizer: PreTrainedTokenizer, infer=False):
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'

        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long) for f in features],
                                 batch_first=True, padding_value=pad)
        attention_mask = pad_sequence([torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
                                      batch_first=True, padding_value=0.)
        # input_length = torch.tensor([f.input_length for f in features], dtype=torch.long)

        if not infer:
            decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                                             batch_first=True, padding_value=pad)
            labels = pad_sequence([torch.tensor(f.labels, dtype=torch.long) for f in features],
                                  batch_first=True, padding_value=-100)
        else:
            decoder_input_ids = torch.tensor([[f.decoder_input_ids[0]] for f in features], dtype=torch.long)
            labels = None

        collated_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
        }

        return collated_input


class Inputter(object):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 max_src_turn=20,
                 max_input_length=256,
                 max_decode_length=64
                 ):
        self.tokenizer = tokenizer
        self.max_src_turn = max_src_turn
        self.max_input_length = max_input_length
        self.max_decode_length = max_decode_length

    def convert_data_to_inputs(self, data):
        process = lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))

        dialog = data['dialog']
        inputs = []
        context = []
        if len(dialog) > self.max_src_turn:
            # only maintain latest `max_src_turn` of dialog history
            dialog = dialog[-self.max_src_turn:]

        logging.info("Model input: {}".format(dialog))

        for i in range(len(dialog)):
            text = _norm(dialog[i]['text'])
            text = process(text)

            if i > 0 and dialog[i]['speaker'] == 'sys':
                res = {
                    'context': context.copy(),
                    'response': text,
                }
                inputs.append(res)
            context = context + [text]
        inputs = inputs[-1:]
        logging.info("inputs: {}".format(inputs))

        return inputs

    def featurize(self, bos, eos, context, max_input_length, response, max_decoder_input_length):
        context = [c + [eos] for c in context]
        input_ids = sum(context, [])[:-1]
        input_ids = input_ids[-max_input_length:]

        labels = (response + [eos])[:max_decoder_input_length]
        decoder_input_ids = [bos] + labels[:-1]

        assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]

        return InputFeature(input_ids, decoder_input_ids, labels)

    def convert_inputs_to_features(self, inputs):
        bos = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
        eos = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

        features = []
        for i in range(len(inputs)):
            ipt = inputs[i]
            feat = self.featurize(
                bos, eos,
                ipt['context'], self.max_input_length,
                ipt['response'], self.max_decode_length,
            )
            features.append(feat)

        return features

    def convert_features_for_model(self, features, infer=True):
        collated_features = FeatureDataset.collate(features, tokenizer=self.tokenizer, infer=infer)
        return collated_features

