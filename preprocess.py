import os
import pickle
import re
import time

import pandas as pd
from gensim.corpora import Dictionary

num_words = 50000
bos = '<BOS>'
bos_id = num_words
eos = '<EOS>'
eos_id = num_words+1
unk = '<UNK>'
unk_id = num_words+2
pad = '<PAD>'
pad_id = num_words+3
num_tokens = num_words+4


def now():
    return time.strftime("%m-%d %H:%M", time.localtime())


def contain_chinese(s):
    if re.findall('[\u4e00-\u9fa5]+', s):
        return True
    return False


def doc2id(doc, token2id):
    for index, word in enumerate(doc):
        if word in token2id:
            doc[index] = token2id[word]
        else:
            doc[index] = token2id[unk]
    return doc


if __name__=='__main__':
    if not os.path.exists('data'):
        os.mkdir('data')

    print(now(), 'Loading Data')
    Q = []
    A = []
    ALL = []
    with open('./data/dgk_shooter_z.conv', 'r', encoding='UTF-8', errors='ignore') as f:
        a = f.readline().strip()
        for line in f:
            q = a
            a = line.strip()
            if a!='E' and contain_chinese(a): ALL.append(a[2:])
            if q=='E' or a=='E' or \
                (not contain_chinese(q)) or \
                (not contain_chinese(a)):
                continue
            Q.append(q[2:])
            A.append(a[2:])

    print(now(), 'Tokenizing')
    Q = [i.split('/') for i in Q]
    A = [i.split('/') for i in A]
    ALL = [i.split('/') for i in ALL]
    corpus = Dictionary(ALL)
    corpus.filter_extremes(no_below=3, no_above=1.0, keep_n=num_words)

    token2id = corpus.token2id
    token2id[bos] = bos_id
    token2id[eos] = eos_id
    token2id[unk] = unk_id
    token2id[pad] = pad_id
    print(now(), 'Dictionary size: {}'.format(len(token2id)))

    Q = [doc2id(word, token2id) for word in Q]
    A = [doc2id(word, token2id) for word in A]

    print(now(), 'Saving Processed Data')
    pickle.dump(Q, open('./data/Q.txt', 'wb'))
    pickle.dump(A, open('./data/A.txt', 'wb'))
    pickle.dump(ALL, open('./data/ALL.txt', 'wb'))
    pickle.dump(token2id, open('./data/token2id.txt', 'wb'))

    print(now(), 'All Done!')
