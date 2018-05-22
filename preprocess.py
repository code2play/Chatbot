import os
import pickle
import re
import time

import pandas as pd
from gensim.corpora import Dictionary


def now():
    return time.strftime("%m-%d %H:%M", time.localtime())


def contain_chinese(s):
    if re.findall('[\u4e00-\u9fa5]+', s):
        return True
    return False


if __name__=='__main__':
    if not os.path.exists('data'):
        os.mkdir('data')

    print(now(), 'Loading Data')
    Q = []
    A = []
    ALL = []
    with open('./data/xiaohuangji50w_fenciA.conv', 'r', encoding='UTF-8', errors='ignore') as f:
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

    token2id = corpus.token2id
    print(now(), 'Dictionary size: {}'.format(len(token2id)))

    print(now(), 'Saving Processed Data')
    pickle.dump(Q, open('./data/Q.txt', 'wb'))
    pickle.dump(A, open('./data/A.txt', 'wb'))
    pickle.dump(token2id, open('./data/token2id.txt', 'wb'))

    print(now(), 'All Done!')
