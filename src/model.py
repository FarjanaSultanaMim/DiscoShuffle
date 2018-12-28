#!/usr/bin/env python
# coding: utf-8

import numpy as np

import keras
from keras.layers import Embedding, LSTM, Conv1D, Input, SpatialDropout1D, Masking, Dense, Average, TimeDistributed, Activation, Softmax, Multiply
from keras.layers import Lambda, Bidirectional, concatenate, Concatenate, Add, GRU
from keras.models import Model, Sequential
from keras.engine.topology import Layer
from keras import backend as K
from keras import initializers

from keras.preprocessing.text import Tokenizer

import data

MAX_PARAGRAPHS = data.MAX_PARAGRAPHS
MAX_VOCAB_WORDS = 45000


def param_str(args):
    keys = [k for k in args.__dict__ if k.startswith("mp_")]
    return "\n".join(["{}={}".format(k[3:], args.__dict__[k]) for k in sorted(keys)])

    
def AttentionalSum(x):
    att = Dense(1, activation="tanh", name="att_layer")(x)
    att = Softmax(axis=1)(att)
    x = Multiply()([att, x])
    return Lambda(lambda _x: K.sum(_x, axis=1))(x)

        
def MeanOverTime():
    return Lambda(lambda x: K.mean(x, axis=1), name='meanOverTime')


def MaxOverTime():
    return Lambda(lambda x: K.max(x, axis=1), name='maxOverTime')


def create_enc_nea(model_input, pre_embed, word_index_m, sequence_length_main, args):
    vocabulary_size_m = min(len(word_index_m) + 1, MAX_VOCAB_WORDS)

    if args.mp_pretrained and pre_embed != None:
        embedding_matrix_m = np.zeros((vocabulary_size_m, args.mp_emb_dim))

        for word, i in word_index_m.items():
            if i >= MAX_VOCAB_WORDS:
                continue
            try:
                embedding_vector_m = pre_embed[word]
                embedding_matrix_m[i-1] = embedding_vector_m
            except KeyError:
                embedding_matrix_m[i-1] = np.random.normal(0, np.sqrt(0.25), args.mp_emb_dim)

    model = Embedding(input_dim = vocabulary_size_m, #embedding_matrix_m.shape[0],
                      output_dim = args.mp_emb_dim, #embedding_matrix_m.shape[1],
                      input_length = sequence_length_main,
                      weights=[embedding_matrix_m] if args.mp_pretrained  and pre_embed != None else None,
                      mask_zero=True,
                      trainable=not args.mp_emb_fix,
                      name='embedding_layer')(model_input)
    
    # Attentional aggregation?
    if args.mp_att:
        model = GRU(args.mp_aggr_grudim, name='att_GRU_layer', dropout=args.mp_dropout, return_sequences=True, trainable=not args.mp_enc_fix)(model)
        model = AttentionalSum(model)
        
    else:
        
        # Mean Over Time or pure.
        if args.mp_mot:
            model = Bidirectional(LSTM(args.mp_aggr_grudim, name='mot_GRU_layer', dropout=args.mp_dropout, return_sequences=True, trainable=not args.mp_enc_fix))(model)
            model = MeanOverTime()(model)

        else:
            model = Bidirectional(
                LSTM(args.mp_encdim, name= 'LSTM_layer', dropout=args.mp_dropout, trainable=not args.mp_enc_fix))(model)
        
    return model


def create_regression(pre_embed, word_index_m, sequence_length_main, sequence_length_pseq, args):
    x = []
    
    x_essay = Input(shape=(sequence_length_main,))
    x += [x_essay]
    
    if args.mp_model_type == "nea":
        model = create_enc_nea(x_essay, pre_embed, word_index_m, sequence_length_main, args)

    if args.mp_pseq:
        x_pseq = Input(shape=(sequence_length_pseq,))
        x += [x_pseq]
        
        y = Embedding(input_dim=5, output_dim=args.mp_pseq_embdim, input_length=sequence_length_pseq, mask_zero=True, name="pseq_embedding_layer")(x_pseq)
        y = LSTM(args.mp_pseq_encdim, name="pseq_LSTM_layer", dropout=args.mp_dropout)(y)
        
        model = Concatenate()([model, y])
        
    model = Dense(1, activation='sigmoid', name="RegressionLayer")(model)

    return Model(inputs=x, outputs=[model])


def create_enc_for_pretrain(pre_embed, word_index_m, sequence_length_main, args):
    model_input = Input(shape=(sequence_length_main,))

    if args.mp_model_type == "nea":
        model = create_enc_nea(model_input, pre_embed, word_index_m, sequence_length_main, args)

    model = Dense(2, activation='softmax', name="CoherenceLayer")(model)

    return Model(inputs=model_input, outputs=[model])


def create_vocab(essays, char_level=False):
    tokenizer_m = Tokenizer(num_words=MAX_VOCAB_WORDS, char_level=char_level, lower=True, oov_token='UNK')
    tokenizer_m.fit_on_texts(essays)
    word_index_m = tokenizer_m.word_index
    
    print('Found %s unique tokens.' % len(word_index_m))    
    
    return tokenizer_m