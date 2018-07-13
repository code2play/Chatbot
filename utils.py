import pickle

bos = '<BOS>'
eos = '<EOS>'
unk = '<UNK>'
pad = '<PAD>'

def load_dict():
    token2id = pickle.load(open('./data/token2id.txt', 'rb'))

    num_words = len(token2id)
    bos_id = num_words
    eos_id = num_words+1
    unk_id = num_words+2
    pad_id = num_words+3
    num_tokens = num_words+4

    token2id[bos] = bos_id
    token2id[eos] = eos_id
    token2id[unk] = unk_id
    token2id[pad] = pad_id

    id2token = {v:k for k, v in token2id.items()}
    return token2id, id2token, num_tokens


def load_QA():
    Q = pickle.load(open('./data/Q.txt', 'rb'))
    A = pickle.load(open('./data/A.txt', 'rb'))
    return Q, A


def load_ALL():
    ALL = pickle.load(open('./data/ALL.txt', 'rb'))
    return ALL

