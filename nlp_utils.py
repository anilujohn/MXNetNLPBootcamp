import os
import pandas as pd
import numpy as np
import collections
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, auc

import time, re
import multiprocessing as mp
import itertools
from tqdm import tqdm, tqdm_notebook
import mxnet as mx
import spacy
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'


# !pip install nltk --user
# !pip install tqdm --user
# !pip install gluonnlp --user
# !pip install torchtext spacy --user
# !python -m spacy download en
# !python -m spacy download de
# ! pip install bokeh --user
# ! pip install xgboost --user

def preprocess():
    
    MAX_SENTENCE_LENGTH = 20
    MAX_VOCAB = 10000
    
    nlp = spacy.load("en")

    word_freq = collections.Counter()
    max_len = 0
    num_rec = 0
    data_location = 'data/umich-sentiment-train.txt'
    print('Count words and build vocab...')
    with open(data_location, 'rb') as f:
        for line in f:
            _lab, _sen = line.decode('utf8').strip().split('\t')
            words = [token.lemma_ for token in nlp(_sen) if token.is_alpha]
            if len(words) > max_len:
                max_len = len(words)
            for word in words:
                word_freq[word] += 1
            num_rec += 1

    # most_common output -> list
    word2idx = {x[0]: i+2 for i, x in enumerate(word_freq.most_common(MAX_VOCAB - 2))}
    word2idx ['PAD'] = 0
    word2idx['UNK'] = 1

    idx2word= {i:v for v, i in word2idx.items()}
    vocab_size = len(word2idx)

    print('Prepare data...')
    y = []
    x = []
    origin_txt = []
    with open(data_location, 'rb') as f:
        for line in f:
            _label, _sen = line.decode('utf8').strip().split('\t')
            origin_txt.append(_sen)
            y.append(int(_label))
            words = [token.lemma_ for token in nlp(_sen) if token.is_alpha] 
            words = [x for x in words if x != '-PRON-']
            _seq = []
            for word in words:
                if word in word2idx.keys():
                    _seq.append(word2idx[word])
                else:
                    _seq.append(word2idx['UNK'])
            if len(_seq) < MAX_SENTENCE_LENGTH:
                _seq.extend([0] * ((MAX_SENTENCE_LENGTH) - len(_seq)))
            else:
                _seq = _seq[:MAX_SENTENCE_LENGTH]
            x.append(_seq)

    print(pd.DataFrame(y, columns = ['yn']).reset_index().groupby('yn').count().reset_index())
    return (x, y, origin_txt, vocab_size, idx2word, word2idx)