import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Word2vecEmbeddingInputlayer
from tensorlayer.nlp import generate_skip_gram_batch, words_to_word_ids
from tensorlayer.prepro import sequences_add_end_id, sequences_add_start_id

from utils import bos, eos, load_ALL, load_dict, pad, unk

ALL = load_ALL()
token2id, id2token, num_tokens = load_dict()
bos_id = token2id[bos]
eos_id = token2id[eos]
unk_id = token2id[unk]
pad_id = token2id[pad]

ALL = [words_to_word_ids(seq, token2id, unk_key=unk) for seq in ALL]
ALL = sequences_add_start_id(ALL, start_id=bos_id)
ALL = sequences_add_end_id(ALL, end_id=eos_id)
ALL = [word for seq in ALL for word in seq]

# Parameters
batch_size = 20  # Note: small batch_size need more steps for a Epoch
emb_dim = 300
skip_window = 5
num_skips = 10
num_sampled = 100
lr = 0.2
n_epoch = 10
num_steps = int((len(ALL) / batch_size) * n_epoch)

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
            
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
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
