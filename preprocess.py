import os
import pickle

from tensorlayer.nlp import create_vocab, words_to_word_ids

bos = '<S>'
eos = '</S>'
unk = '<UNK>'
pad = '<PAD>'
vocab_dir = './data/vocab.txt'

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.mkdir('data')

    Q = []
    A = []
    ALL = []
    with open('./data/xiaohuangji50w_fenciA.conv', 'r', encoding='UTF-8') as f:
        a = f.readline().strip()
        for line in f:
            q = a
            a = line.strip()
            if a!='E':
                a = [bos] + a[2:].split('/') + [eos]
                ALL.append(a)
            if q=='E' or a=='E':
                continue
            # 不需要bos
            Q.append(q[1:])
            A.append(a[1:])

    vocab = create_vocab(ALL, vocab_dir, min_word_count=3)

    token2id = vocab._vocab
    unk_id = vocab._unk_id
    token2id[unk] = unk_id
    
    Q = [words_to_word_ids(s, token2id, unk_key=unk) for s in Q]
    A = [words_to_word_ids(s, token2id, unk_key=unk) for s in A]
    ALL = [words_to_word_ids(s, token2id, unk_key=unk) for s in ALL]

    pickle.dump(Q, open('./data/Q.txt', 'wb'))
    pickle.dump(A, open('./data/A.txt', 'wb'))
    pickle.dump(ALL, open('./data/ALL.txt', 'wb'))

    print('All Done!')
