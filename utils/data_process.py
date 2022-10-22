# coding=utf-8
import gzip
import json
import os
import math
import random
import pickle
from functools import partial
from math import ceil
from torch.utils.data import DataLoader, Sampler
import tqdm
from utils.inputter_utils import FeatureDataset
from transformers.tokenization_utils import PreTrainedTokenizer


def _norm(s):
    return ' '.join(s.strip().split())


class BucketSampler(Sampler):
    """
    this sampler will sort data by sequence length
    """

    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i + self._bucket_size],
                          key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i + self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s // self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s / self._batch_size) for s in bucket_sizes)


class BucketingDataLoader(object):
    def __init__(self, toker, feature_dataset, batch_size,
                 bucket=100, shuffle=True, **kwargs):
        with open(f'./DATA/data.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.toker = toker
        self.feature_dataset = feature_dataset
        self.batch_size = batch_size
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle

    def __iter__(self):
        trunc_chunk = []
        lens = []
        for feat in self.data:
            trunc_chunk.append(feat)
            lens.append(feat.input_len)

        dataset = self.feature_dataset(trunc_chunk)
        sampler = BucketSampler(lens, self.bucket_size, self.batch_size,
                                droplast=True, shuffle=self.shuffle)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=0,  # can test multi-worker
                            collate_fn=partial(self.feature_dataset.collate, tokenizer=self.toker))
        yield from loader

    def __len__(self):
        return len(self.data)


class DistributedBucketingDataLoader(BucketingDataLoader):
    """ distributed version """

    def __init__(self, rank, num_replica, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.num_replica = num_replica
        self.data = self.data[self.rank::self.num_replica]


class DynamicBatchingLoader(object):
    """ this loader takes raw text file, used for validate perplexity """

    def __init__(self, corpus_file, toker, batch_size, **kwargs):
        self.corpus = corpus_file
        self.toker = toker
        self.bs = batch_size
        self.num_examples = self.get_len(corpus_file)
        self.kwargs = kwargs

    def __iter__(self, epoch=1):
        if epoch > 0:
            for epoch in range(epoch):
                yield from self._iter_epoch()
        else:
            while True:
                yield from self._iter_epoch()

    def __len__(self):
        return ceil(self.num_examples / self.bs)

    def _iter_epoch(self):
        try:
            with open(self.corpus, 'r', encoding="utf-8") as f:
                reader = f.readlines()

            features = []
            for line in tqdm.tqdm(reader, total=len(reader), desc=f"validating"):
                data = json.loads(line)
                inputs = convert_data_to_inputs(data, self.toker, **self.kwargs)
                features.extend(convert_inputs_to_features(inputs, self.toker, **self.kwargs))
                if len(features) >= self.bs:
                    batch = self._batch_feature(features)
                    yield batch
                    features = []

            if len(features) > 0:
                batch = self._batch_feature(features)
                yield batch

        except StopIteration:
            pass

    def _batch_feature(self, features):
        return FeatureDataset.collate(features, self.toker)

    def get_len(self, corpus):
        with open(corpus, 'r', encoding="utf-8") as file:
            reader = [json.loads(line) for line in file]
        return sum(map(lambda x: len(list(filter(lambda y: y['speaker'] == 'sys', x['dialog'][1:]))), reader))


def convert_data_to_inputs(data, toker: PreTrainedTokenizer, **kwargs):
    process = lambda x: toker.convert_tokens_to_ids(toker.tokenize(x))

    dialog = data['dialog']
    inputs = []
    context = []

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

    return inputs


def convert_inputs_to_features(inputs, toker, **kwargs):
    if len(inputs) == 0:
        return []

    assert kwargs.get('max_input_length', None) is not None, 'you should give max_input_length'
    max_input_length = kwargs.get('max_input_length')
    assert kwargs.get('max_decoder_input_length', None) is not None, 'you should give max_decoder_input_length'
    max_decoder_input_length = kwargs.get('max_decoder_input_length')

    pad = toker.pad_token_id
    if pad is None:
        pad = toker.eos_token_id
        assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
    bos = toker.bos_token_id
    if bos is None:
        bos = toker.cls_token_id
        assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
    eos = toker.eos_token_id
    if eos is None:
        eos = toker.sep_token_id
        assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

    features = []
    for i in range(len(inputs)):
        ipt = inputs[i]
        feat = featurize(
            bos, eos,
            ipt['context'], max_input_length,
            ipt['response'], max_decoder_input_length,
        )
        features.append(feat)
    return features


class InputFeatures(object):
    def __init__(
            self,
            input_ids,
            decoder_input_ids, labels,
    ):
        self.input_ids = input_ids
        self.input_length = len(input_ids)

        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels

        self.input_len = self.input_length + self.decoder_input_length


def featurize(
        bos, eos,
        context, max_input_length,
        response, max_decoder_input_length,
):
    context = [c + [eos] for c in context]
    input_ids = sum(context, [])[:-1]
    input_ids = input_ids[-max_input_length:]

    labels = (response + [eos])[:max_decoder_input_length]
    decoder_input_ids = [bos] + labels[:-1]

    assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]

    return InputFeatures(
        input_ids,
        decoder_input_ids, labels,
    )
