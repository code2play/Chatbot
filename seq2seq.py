import json

import pandas as pd
from keras.layers import (LSTM, Bidirectional, CuDNNLSTM, Dense, Embedding,
                          Input, RepeatVector)
from keras.layers.wrappers import TimeDistributed
from keras.metrics import top_k_categorical_accuracy
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from preprocess import EOS, maxlen, num_words

Q = pd.read_csv('./data/Q.csv', index_col=0)
A = pd.read_csv('./data/A.csv', index_col=0)

token2id = json.load(open('./data/token2id.json', 'r'))
id2token = json.load(open('./data/id2token.json', 'r'))

X_train, X_test, y_train, y_test = train_test_split(Q, A, 
                                        test_size=0.01, random_state=42)

target = y_train.apply(lambda x: x[1:])
embedding = Sequential([
    Embedding(input_dim=num_words, 
              output_dim=300, 
              input_length=maxlen)
])
embedding.compile('adam', 'mse')
traget = embedding.predict(fuck)

encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=num_words, 
                              output_dim=300, 
                              input_length=maxlen)(encoder_inputs)
encoder = LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
dncoder_embedding = Embedding(input_dim=num_words, 
                              output_dim=300, 
                              input_length=maxlen)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True)
decoder_outputs = decoder_lstm(dncoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit([X_train, y_train], fuck,
          batch_size=128,
          epochs=10)
