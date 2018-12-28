
import argparse, logging

import pickle

import pandas as pd
import numpy as np
from collections import Counter
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers
from keras.callbacks import ModelCheckpoint

import hashlib

import tensorflow as tf

import model
import data
import kr_util


def main(args):
    pstr = model.param_str(args)
    out_dir = hashlib.md5(pstr.encode("utf-8")).hexdigest()
    out_dir = os.path.join("output_enc", out_dir)
    os.system("mkdir -p {}".format(out_dir))
    
    with open(os.path.join(out_dir, "param.txt"), "w") as f:
        print(pstr, file=f)

    print("")
    print("Essay encoder trainer")
    print("  # Output dir: {}".format(out_dir))
    print("  # Param string:\n{}".format(pstr))
    print("")
        
    # To reduce memory consumption
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    
    # Setup data
    pre_embed = data.load_pretrained_embeddings() if args.mp_pretrained else None
    essays = data.load_essay('/home/mim/ICLE_all_data.csv')
    essays = data.preprocess_essay(essays)
    
    if args.mp_shuf == "sentence":
        main_essay, main_scores = data.create_training_data_for_shuffled_essays(essays)
        
    elif args.mp_shuf == "di":
        di_list = data.load_discourse_indicators()
        main_essay, main_scores = data.create_training_data_for_di_shuffled_essays(essays, di_list)

    main_essay_t, main_essay_v, score_t, score_v = train_test_split(
        main_essay, main_scores,
        test_size=0.2, shuffle=True, random_state=33)

    # Create a tokenizer (indexer) from the training dataset.
    tokenizer_m = model.create_vocab(main_essay_t)
    
    with open(os.path.join(out_dir, "tokenizer.pickle"), "wb") as f:
        pickle.dump(tokenizer_m, f)

    sequences_train_main = tokenizer_m.texts_to_sequences(main_essay_t)
    sequences_valid_main = tokenizer_m.texts_to_sequences(main_essay_v)
    lens = [len(e) for e in sequences_train_main]

    X_train_main = pad_sequences(sequences_train_main, maxlen=min(max(lens), data.MAX_WORDS))
    X_val_main = pad_sequences(sequences_valid_main, maxlen=X_train_main.shape[1])

    sequence_length_main = X_train_main.shape[1]

    mainModel = model.create_enc_for_pretrain(pre_embed,
                                            tokenizer_m.word_index,
                                            sequence_length_main,
                                            args,
                                            )
    mainModel.summary()


    optimizer_main=keras.optimizers.Adam(clipnorm=args.mp_clipnorm)
    mainModel.compile(optimizer=optimizer_main, loss='categorical_crossentropy', metrics=['accuracy'])
    
    score_t_c = keras.utils.to_categorical(score_t)
    score_v_c = keras.utils.to_categorical(score_v)

    es=keras.callbacks.EarlyStopping(monitor='val_loss',
                                     min_delta=0,
                                     patience=7,
                                     verbose=1, mode='auto', baseline=None,
                                     restore_best_weights=True)

    nbl = kr_util.NBatchLogger(out_dir)

    mainModel.fit(X_train_main, score_t_c, validation_data=(X_val_main, score_v_c),
                epochs=100, verbose=1, callbacks=[es, nbl],
              batch_size=32)

    mainModel.save_weights(os.path.join(out_dir, "encoder.hdf5"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m','--model-type', dest='mp_model_type', type=str, required=True,
        help="Type of model (nea, rnn1, rnn2).")
    parser.add_argument(
        '-d','--dropout', dest='mp_dropout', type=float, required=True,
        help="Dropout ratio.") 
    parser.add_argument(
        '-p','--pre-trained', dest='mp_pretrained', action="store_true",
        help="Whether to use pretrained embeddings.")
    parser.add_argument(
        '-f','--fix-embedding', dest='mp_emb_fix', action="store_true",
        help="Whether to fix word embeddings.")
    parser.add_argument(
        '-fe','--fix-encoder', dest='mp_enc_fix', action="store_true",
        help="Whether to fix encoder.")
    parser.add_argument(
        '-ed','--embedding-dim', dest='mp_emb_dim', type=int, required=True,
        help="Dimension of word embeddings.")
    parser.add_argument(
        '-aggrgru','--aggregation-grudim', dest='mp_aggr_grudim', type=int,
        help="Dimension of GRU encoder for aggregation.")    
    parser.add_argument(
        '-encd','--encoder-dim', dest='mp_encdim', type=int,
        help="Dimension of LSTM encoder.")        
    parser.add_argument(
        '-mot','--meanovertime', dest='mp_mot', action="store_true",
        help="Whether to use MOT layer ot nor.")
    parser.add_argument(
        '-att','--attention', dest='mp_att', action="store_true",
        help="Whether to use final attention layer ot nor.")    
    parser.add_argument(
        '-gc','--gradientclipnorm', dest='mp_clipnorm', type=float, required=True,
        help="Gradient clipping norm.") 
    
    parser.add_argument(
        '-shuf','--shuffle-type', dest='mp_shuf', type=str, required=True,
        help="Shuffling type (sentence, di).")
    args = parser.parse_args()

    main(args)
