import json
import os
import re
import time

import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

EOS = 'EOS'
UNK = 'UNK'
num_words = 100000
maxlen = 15


def now():
    return time.strftime("%m-%d %H:%M", time.localtime())


def contain_chinese(s):
    if re.findall('[\u4e00-\u9fa5]+', s):
        return True
    return False


def to_seq(tokenizer, data, maxlen):
    data = tokenizer.texts_to_sequences(data)
    data = pad_sequences(data, maxlen=maxlen)
    data = pd.DataFrame(data)
    return data


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
            if a!='E' and contain_chinese(a): ALL.append(a[2:] + '/{}'.format(EOS))
            if q=='E' or a=='E' or \
                (not contain_chinese(q)) or \
                (not contain_chinese(a)):
                continue
            Q.append(q[2:] + '/{}'.format(EOS))
            A.append(a[2:] + '/{}'.format(EOS))

    print(now(), 'Tokenizing')
    tokenizer = Tokenizer(num_words=num_words, split='/', lower=False)
    tokenizer.fit_on_texts(ALL)

    Q = to_seq(tokenizer, Q, maxlen)
    Q.to_csv('./data/Q.csv')
    A = to_seq(tokenizer, A, maxlen)
    A.to_csv('./data/A.csv')

    print(now(), 'Saving Dictionary')
    token2id = tokenizer.word_index
    json.dump(token2id, open('./data/token2id.json', 'w'))
    id2token = {v:k for k,v in token2id.items()}
    json.dump(id2token, open('./data/id2token.json', 'w'))

    print(now(), 'All Done!')
