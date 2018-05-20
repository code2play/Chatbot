import pickle
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
from sklearn.model_selection import train_test_split
from tensorlayer.cost import cross_entropy_seq_with_mask
from tensorlayer.layers import *
from tensorlayer.prepro import (pad_sequences, sequences_add_end_id,
                                sequences_add_start_id, sequences_get_mask)

from preprocess import (bos, bos_id, eos, eos_id, now, num_tokens, num_words,
                        pad, pad_id, unk, unk_id)

Q = pickle.load(open('./data/Q.txt', 'rb'))
A = pickle.load(open('./data/A.txt', 'rb'))

token2id = pickle.load(open('./data/token2id.txt', 'rb'))
id2token = {v:k for k, v in token2id.items()}

Q = sequences_add_end_id(Q, end_id=eos_id)
A = sequences_add_end_id(A, end_id=eos_id)

X_train, X_test, y_train, y_test = train_test_split(Q, A, 
                                        test_size=0.01, random_state=42)

# parameters
batch_size = 32
n_step = np.ceil(len(X_train)/batch_size)
emb_dim = 500
lr = 0.01
n_epoch = 50

def model(encode_in, decode_in, isTrain=True, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        def embedding(seq):
            with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
                emb = EmbeddingInputlayer(
                    inputs=seq,
                    vocabulary_size=num_tokens,
                    embedding_size=emb_dim,
                    name='embedding_seq'
                )
            return emb

        seq2seq = Seq2Seq(
            net_encode_in=embedding(encode_in),
            net_decode_in=embedding(decode_in),
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=emb_dim,
            encode_sequence_length=retrieve_seq_length_op2(encode_in),
            decode_sequence_length=retrieve_seq_length_op2(decode_in),
            dropout=(0.5 if isTrain else None),
            n_layer=3,
            return_seq_2d=True,
            name='seq2seq'
        )
        net_out = DenseLayer(seq2seq, n_units=num_tokens, act=tf.identity, name='output')
        return net_out, seq2seq

# Train tensors
encode_in = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='encode_in')
decode_in = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='decode_in')
target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask")
net_out, _ = model(encode_in, decode_in, isTrain=True, reuse=False)

# Test tensors
# encode_in_1d = tf.placeholder(dtype=tf.int64, shape=[1, None], name='encode_in')
# decode_in_1d = tf.placeholder(dtype=tf.int64, shape=[1, None], name='decode_in')
# net, seq2seq = model(encode_in_1d, decode_in_1d, isTrain=False, reuse=True)
# y = tf.nn.softmax(net.outputs)

loss = cross_entropy_seq_with_mask(
    logits=net_out.outputs, 
    target_seqs=target_seqs,
    input_mask=target_mask,
    name='loss'
)
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)
# tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=net)

for epoch in range(n_epoch):
    epoch_time = time.time()
    n_iter, total_err = 0, 0
    for X, y in tl.iterate.minibatches(X_train, y_train, batch_size=batch_size, shuffle=False):
        step_time = time.time()
        
        batch_encode_in = pad_sequences(X)
        batch_decode_in = sequences_add_start_id(y, start_id=bos_id, remove_last=False)
        batch_decode_in = pad_sequences(batch_decode_in)
        batch_target_seqs = sequences_add_end_id(y, end_id=eos_id)
        batch_target_seqs = pad_sequences(batch_target_seqs)
        batch_target_mask = sequences_get_mask(batch_target_seqs)

        _, err = sess.run([train_op, loss], {
                           encode_in: batch_encode_in,
                           decode_in: batch_decode_in,
                           target_seqs: batch_target_seqs,
                           target_mask: batch_target_mask})

        if n_iter % 200 == 0:
            print("Epoch[%d/%d] step:[%d/%d] loss:%f took:%.5fs" % \
                 (epoch, n_epoch, n_iter, n_step, err, time.time() - step_time))

        total_err += err; n_iter += 1
