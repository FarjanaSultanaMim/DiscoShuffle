#!/usr/bin/env python
# coding: utf-8

import numpy as np

import keras
from keras.layers import Embedding, LSTM, Conv1D, Input, SpatialDropout1D, Masking, Dense, Average, TimeDistributed, Activation, Softmax, Multiply
from keras.layers import Lambda, Bidirectional, concatenate, Concatenate, Add, GRU, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model, Sequential
from keras import backend as K
from keras import initializers
import tensorflow as tf
import tensorflow_hub as hub

import keras.layers as layers
from keras.models import load_model
from keras.engine import Layer

from keras.preprocessing.text import Tokenizer

import data

MAX_PARAGRAPHS = data.MAX_PARAGRAPHS
MAX_VOCAB_WORDS = 15000
MAX_VOCAB_WORDS_enc = 90000
MAX_PROMPT_WORDS = 1000
batch_size = 32

elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


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

def MOT():
    return Lambda(lambda x: K.mean(x, axis=1), name='MOT')


def MaxOverTime():
    return Lambda(lambda x: K.max(x, axis=1), name='maxOverTime')


def create_enc_nea(model_input, pre_embed, word_index_m, sequence_length_main, args, for_pretrain=False):
    if for_pretrain:
        vocabulary_size_m = min(len(word_index_m) + 1, MAX_VOCAB_WORDS_enc)
    
    else:
        vocabulary_size_m = min(len(word_index_m) + 1, MAX_VOCAB_WORDS)

    if args.mp_pretrained and pre_embed != None:
        embedding_matrix_m = np.zeros((vocabulary_size_m, args.mp_emb_dim))

        for word, i in word_index_m.items():
            if i >= MAX_VOCAB_WORDS:
                continue
            try:
                embedding_vector_m = pre_embed[word]
                embedding_matrix_m[i] = embedding_vector_m
            except KeyError:
                embedding_matrix_m[i] = np.random.normal(0, np.sqrt(0.25), args.mp_emb_dim)

    model = Embedding(input_dim = vocabulary_size_m, #embedding_matrix_m.shape[0],
                      output_dim = args.mp_emb_dim, #embedding_matrix_m.shape[1],
                      input_length = sequence_length_main,
                      weights=[embedding_matrix_m] if args.mp_pretrained  and pre_embed != None else None,
                      mask_zero=True,
                      trainable=not args.mp_emb_fix,
                      name='embedding_layer')(model_input)
    
    # Attentional aggregation?

    if args.mp_att:
        model = LSTM(args.mp_aggr_grudim, name='att_GRU_layer', dropout=args.mp_dropout, return_sequences=True, trainable=not args.mp_enc_fix)(model)
        model = AttentionalSum(model)
    
    else:
        
        # Mean Over Time or pure.
        if args.mp_mot:
            
            model = Bidirectional(LSTM(args.mp_aggr_grudim, dropout=args.mp_dropout, return_sequences=True, trainable=not args.mp_enc_fix),  name= 'MOT_bi-LSTM_layer')(model)
            model = MeanOverTime()(model)
        
        else:
            model = LSTM(args.mp_aggr_grudim, name= 'LSTM_layer', dropout=args.mp_dropout, trainable=not args.mp_enc_fix, return_sequences=True)(model)
            model = MeanOverTime()(model)
        
    return model

def create_enc_nea_elmo(model_input, args):
    
    
    model = ElmoEmbeddingLayer(trainable=False)(model_input)
    print(model)
    
    # Attentional aggregation?
    if args.mp_att:
        model = LSTM(args.mp_aggr_grudim, name='att_GRU_layer', dropout=args.mp_dropout, return_sequences=True)(model)
        model = AttentionalSum(model)
        
    else:
        
        # Mean Over Time or pure.
        if args.mp_mot:
            model = LSTM(
                args.mp_aggr_grudim, name='mot_GRU_layer', dropout=args.mp_dropout, return_sequences=True)(model)
            model = MeanOverTime()(model)

        else:
            model = Bidirectional(
                LSTM(args.mp_encdim, name= 'LSTM_layer', dropout=args.mp_dropout))(model)
        
    return model

def create_enc_prompt(model_input, word_index_p, sequence_length_prompt, args):
    
    pre_embed = data.load_pretrained_embeddings() 
    
    vocabulary_size_p = min(len(word_index_p) + 1, MAX_PROMPT_WORDS)

    embedding_matrix_p = np.zeros((vocabulary_size_p, 50))

    for word, i in word_index_p.items():
        if i >= MAX_PROMPT_WORDS:
            continue
        try:
            embedding_vector_p = pre_embed[word]
            embedding_matrix_p[i] = embedding_vector_p
        except KeyError:
            embedding_matrix_p[i] = np.random.normal(0, np.sqrt(0.25), 50)

    model = Embedding(input_dim = vocabulary_size_p, #embedding_matrix_m.shape[0],
                      output_dim = 50, #embedding_matrix_m.shape[1],
                      input_length = sequence_length_prompt,
                      weights=[embedding_matrix_p], 
                      mask_zero=True,
                      trainable= True,
                      name='prompt_embedding_layer')(model_input)
    
  
    model = LSTM (300, dropout=args.mp_dropout, name='prompt_LSTM_layer')(model)
        
    return model


def create_regression(pre_embed, word_index_m, sequence_length_main, sequence_length_pseq, args):
    x = []
    
    x_essay = Input(shape=(sequence_length_main,))
    x += [x_essay]
    
    if args.mp_model_type == "nea":
        model = create_enc_nea(x_essay, pre_embed, word_index_m, sequence_length_main, args)
        
    if args.mp_model_type == "nea_aft_pretrain":
        model = create_enc_nea(x_essay, pre_embed, word_index_m, sequence_length_main, args, for_pretrain=True)

    if args.mp_pseq:
        x_pseq = Input(shape=(sequence_length_pseq,))
        x += [x_pseq]
        
        y = Embedding(input_dim=4, output_dim=args.mp_pseq_embdim, input_length=sequence_length_pseq, mask_zero=True, name="pseq_embedding_layer")(x_pseq)
        y = Bidirectional(LSTM(args.mp_pseq_encdim, dropout=args.mp_dropout), name="pseq_LSTM_layer")(y)
        
        model = Concatenate()([model, y])
      
    model = Dense(1, activation='sigmoid', name="RegressionLayer")(model)

    return Model(inputs=x, outputs=[model])

def create_regression_wprompt(pre_embed, word_index_m, word_index_p, sequence_length_main, sequence_length_pseq, sequence_length_prompt, args):
    
    x = []
    
    x_essay = Input(shape=(sequence_length_main,))
    x += [x_essay]
    
    if args.mp_model_type == "nea":
        model = create_enc_nea(x_essay, pre_embed, word_index_m, sequence_length_main, args)
        
    if args.mp_model_type == "nea_aft_pretrain":
        model = create_enc_nea(x_essay, pre_embed, word_index_m, sequence_length_main, args, for_pretrain=True)
        
    if args.mp_pseq:
        x_pseq = Input(shape=(sequence_length_pseq,))
        x += [x_pseq]
        
        y = Embedding(input_dim=4, output_dim=args.mp_pseq_embdim, input_length=sequence_length_pseq, mask_zero=True, name="pseq_embedding_layer")(x_pseq)
        y = Bidirectional(LSTM(args.mp_pseq_encdim, dropout=args.mp_dropout), name="pseq_LSTM_layer")(y)
        
        model = Concatenate()([model, y])
        
    if args.mp_prompt:
        
        x_prompt = Input(shape=(sequence_length_prompt,))
        x += [x_prompt]
        
        p = create_enc_prompt(x_prompt, word_index_p, sequence_length_prompt, args)
        
        model = Concatenate()([model, p])

    model = Dense(1, activation='sigmoid', name="RegressionLayer")(model)

    return Model(inputs=x, outputs=[model])

def create_regression_elmo(sequence_length_main, sequence_length_pseq, args):
    
    x = []
    
    x_essay = Input(shape=(sequence_length_main,), dtype="string")
    x += [x_essay]
    
    if args.mp_model_type == "nea":
        model = create_enc_nea_elmo(x_essay, args)
        
    if args.mp_model_type == "nea_aft_pretrain":
        model = create_enc_nea_elmo(x_essay, args, for_pretrain=True)
    
    if args.mp_pseq:
        x_pseq = Input(shape=(sequence_length_pseq,))
        x += [x_pseq]
        
        y = Embedding(input_dim=4, output_dim=args.mp_pseq_embdim, input_length=sequence_length_pseq, mask_zero=True, name="pseq_embedding_layer")(x_pseq)
        y = Bidirectional(LSTM(args.mp_pseq_encdim, name="pseq_LSTM_layer", dropout=args.mp_dropout))(y)
        
        model = Concatenate()([model, y])
      
    model = Dense(1, activation='sigmoid', name="RegressionLayer")(model)

    return Model(inputs=x, outputs=[model])

def pseq_regression(sequence_length_pseq, args):
    x = []

    x_pseq = Input(shape=(sequence_length_pseq,))
    x += [x_pseq]
        
    y = Embedding(input_dim=4, output_dim=args.mp_pseq_embdim, input_length=sequence_length_pseq, name="pseq_embedding_layer")(x_pseq)
    y = Bidirectional(LSTM(args.mp_pseq_encdim, name="pseq_LSTM_layer", dropout=args.mp_dropout))(y)
        
    model = Dense(1, activation='sigmoid', name="RegressionLayer")(y)

    return Model(inputs=x, outputs=[model])


def create_enc_for_pretrain(pre_embed, word_index_m, sequence_length_main, args):
    model_input = Input(shape=(sequence_length_main,))

    if args.mp_model_type == "nea":
        model = create_enc_nea(model_input, pre_embed, word_index_m, sequence_length_main, args, for_pretrain=True)

    model = Dense(2, activation='softmax', name="CoherenceLayer")(model)

    return Model(inputs=model_input, outputs=[model])


def create_vocab(essays, args, char_level=False):
    
    if args.mp_punct:
        tokenizer_m = keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_WORDS,filters='', char_level=char_level, lower=True, oov_token='UNK')
        
    else:
        tokenizer_m = keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_WORDS, char_level=char_level, lower=True, oov_token='UNK')
    tokenizer_m.fit_on_texts(essays)
    word_index_m = tokenizer_m.word_index
    
    print('Found %s unique tokens.' % len(word_index_m))    
    
    return tokenizer_m

def create_vocab_prompt(prompt, args, char_level=False):
    
    if args.mp_punct:
        tokenizer_m = keras.preprocessing.text.Tokenizer(num_words= None, filters='', char_level=char_level, lower=True, oov_token='UNK')
    else:
        tokenizer_m = keras.preprocessing.text.Tokenizer(num_words= None, char_level=char_level, lower=True, oov_token='UNK')
    tokenizer_m.fit_on_texts(prompt)
    word_index_m = tokenizer_m.word_index
    
    print('Found %s unique tokens.' % len(word_index_m))    
    
    return tokenizer_m

def create_vocab_seq(essays, char_level=False):
    tokenizer_m = keras.preprocessing.text.Tokenizer(lower=True, char_level=char_level)
    tokenizer_m.fit_on_texts(essays)
    word_index_m = tokenizer_m.word_index
    
    print('Found %s unique tokens.' % len(word_index_m))    
    
    return tokenizer_m


def create_vocab_encoder(essays, args, char_level=False):
    
    if args.mp_punct:
        tokenizer_m = keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_WORDS_enc, filters='', char_level=char_level, lower=True, oov_token='UNK')
    else:
        tokenizer_m = keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_WORDS_enc, char_level=char_level, lower=True, oov_token='UNK')
    tokenizer_m.fit_on_texts(essays)
    word_index_m = tokenizer_m.word_index
    
    print('Found %s unique tokens.' % len(word_index_m))    
    
    return tokenizer_m

class ElmoEmbeddingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable, name="{}_module".format(self.name))
        
        if self.trainable:
            self.trainable_weights += keras.backend.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
            
        super(ElmoEmbeddingLayer, self).build(input_shape)
        
    def call(self, x, mask=None):
        original_shape_x = keras.backend.shape(x)
        _x_flatten = keras.backend.flatten(x)
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.trainable)
        _x_flatten = keras.backend.cast(_x_flatten, tf.string)
        embeddings = elmo(_x_flatten, signature="default", as_dict=True)["default"]
        return keras.backend.reshape(embeddings, (original_shape_x[0], original_shape_x[1], 1024))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dimensions)
 