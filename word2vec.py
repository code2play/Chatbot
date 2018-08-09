import pickle
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Word2vecEmbeddingInputlayer
from tensorlayer.nlp import Vocabulary, generate_skip_gram_batch

from preprocess import bos, eos, pad, unk, vocab_dir

ALL = pickle.load(open('./data/ALL.txt', 'rb'))

vocab = Vocabulary(vocab_dir, bos, eos, unk, pad)
token2id = vocab.vocab
id2token = vocab.reverse_vocab
num_tokens = len(token2id)
bos_id = vocab.start_id
eos_id = vocab.end_id
unk_id = vocab.unk_id
pad_id = vocab.pad_id

ALL = [word for seq in ALL for word in seq]

# Parameters
batch_size = 20
emb_dim = 300
skip_window = 5
num_skips = 10
num_sampled = 100
n_epoch = 15
num_steps = int((len(ALL) / batch_size) * n_epoch)
lr = 0.2
# starter_learning_rate = 0.1
# global_step = tf.Variable(0)
# lr = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                 10000, 0.99, staircase=True)

train_inputs = tf.placeholder(tf.int32, shape=(batch_size))
train_labels = tf.placeholder(tf.int32, shape=(batch_size, 1))
net = tl.layers.Word2vecEmbeddingInputlayer(
    inputs=train_inputs,
    train_labels=train_labels, 
    vocabulary_size=num_tokens, 
    embedding_size=emb_dim
)
cost = net.nce_cost
train_params = net.all_params
train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost, var_list=train_params)

with tf.Session() as sess:
    tl.layers.initialize_global_variables(sess)
    average_loss = 0
    step = 0
    data_index = 0
    print_freq = 2000
    step_time = time.time()
    while step < num_steps:
        batch_inputs, batch_labels, data_index = tl.nlp.generate_skip_gram_batch(
            data=ALL, 
            batch_size=batch_size, 
            num_skips=num_skips, 
            skip_window=skip_window, 
            data_index=data_index)
            
        feed_dict = {train_inputs: batch_inputs, 
                     train_labels: batch_labels}
        _, loss_val = sess.run([train_op, cost], feed_dict=feed_dict)
        average_loss += loss_val

        if step % print_freq == 0:
            if step > 0:
                average_loss /= print_freq
            print(f"Step[{step}/{num_steps}] loss:{average_loss} "
                  f"took:{(time.time()-step_time):.2f}s")
            step_time = time.time()
            average_loss = 0

        step += 1
    tl.files.save_npz(net.normalized_embeddings, name='word2vec.npz', sess=sess)
