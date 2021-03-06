import argparse, logging

import pickle
import json
import re
import codecs

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
from sklearn.model_selection import train_test_split

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
    
    # Load Pretrained Embeddings
    pre_embed = data.load_pretrained_embeddings() if args.mp_pretrained else None
    
    #Load essay
    df_icle = data.load_essay_xlsx('PATH_to_ICLEessays(.xlsx)')
    essay_icle = data.get_essay_array_pretrain(df_icle, icle=True)
    #print(len(essay_icle))

    df_toefl = data.load_essay_xlsx('PATH_to_TOEFL11essays(.xlsx)')
    essay_toefl = data.get_essay_array_pretrain(df_toefl)

    # Select which essays to use
    if args.mp_essay_selection == 'ICLEandTOEFL11': 
        essays = np.concatenate((essay_icle, essay_toefl), axis=0)
    elif args.mp_essay_selection == 'AllEssay':
        df_asap = data.load_essay_xlsx('PATH_to_ASAPessays(.xlsx)')
        essay_asap = data.get_essay_array_pretrain(df_asap)

        df_icnale = data.load_essay_xlsx('PATH_to_ICNALEessays(.xlsx)') 
        essay_icnale = data.get_essay_array_pretrain(df_icnale)
        essay_icnale = [re.sub('\ufeff', '', e) for e in essay_icnale]

        essays = np.concatenate((essay_icle, essay_toefl, essay_asap, essay_icnale), axis=0)
    else:
        essays = essay_icle

          
    # Preprocess Essays
    
    if args.mp_wPara:
        essays = data.preprocess_essay_withParaBoundary(essays, args, boseos=True)
    else:
        essays = data.preprocess_essay_noParaBoundary(essays, args)
        
    essays = essays.tolist()
    print("Length of all essays")
    print(len(essays))
    
    if args.mp_shuf == "sentence":
        main_essay, main_scores = data.create_training_data_for_shuffled_essays(essays)
    elif args.mp_shuf == "di":
        di_list = data.load_discourse_indicators()
        main_essay, main_scores = data.create_training_data_for_di_shuffled_essays(essays, di_list)
    elif args.mp_shuf == "para":
        main_essay, main_scores = data.create_training_data_for_paragraph_shuffled_essays(essays)

    main_essay_t, main_essay_v, score_t, score_v = train_test_split(
        main_essay, main_scores,
        test_size=0.2, shuffle=True, random_state=33)
    
    print(main_essay[0])
    print(main_essay[-1])


    # Create a tokenizer (indexer) from the training dataset.
    if args.mp_preenc != None:
        tokenizer_m = pickle.load(open(os.path.join(args.mp_preenc, "tokenizer.pickle"), "rb"))
    else:
        tokenizer_m = model.create_vocab_encoder(main_essay_t, args)
    
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
    
    if args.mp_preenc != None:
        mainModel.load_weights(os.path.join(args.mp_preenc, "encoder.hdf5"), by_name=True)

    print("Starting training.")


    optimizer_main=keras.optimizers.Adam(clipnorm=args.mp_clipnorm)
    mainModel.compile(optimizer=optimizer_main, loss='categorical_crossentropy', metrics=['accuracy'])
    
    score_t_c = keras.utils.to_categorical(score_t)
    score_v_c = keras.utils.to_categorical(score_v)

    es=keras.callbacks.EarlyStopping(monitor='val_loss',
                                     min_delta=0,
                                     patience=5,
                                     verbose=1, mode='auto', baseline=None,
                                     restore_best_weights=True)

    nbl = kr_util.NBatchLogger(os.path.join(out_dir, "logs.pickle"))

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
        '-u_lstm','--uni-lstm', dest='mp_ulstm', action="store_true",
        help="Whether to use Unidirectional LSTM or nor.")
    parser.add_argument(
        '-att','--attention', dest='mp_att', action="store_true",
        help="Whether to use final attention layer ot nor.")    
    parser.add_argument(
        '-gc','--gradientclipnorm', dest='mp_clipnorm', type=float, required=True,
        help="Gradient clipping norm.") 
    parser.add_argument(
        '-punct','--punctuation', dest='mp_punct', action="store_true",
        help="Whether to use punctuation or not.")
    parser.add_argument(
        '-enc','--pretrained-encoder', dest='mp_preenc', type=str,
        help="Path to pretrained encoder.")
    
    parser.add_argument(
        '-es','--essay-selection', dest='mp_essay_selection', default="ICLE",
        help="Type of essay data (ICLE, ICLEandTOEFL11, AllEssay).")
    parser.add_argument(
        '-wpara','--w-para', dest='mp_wPara', action="store_true",
        help="Whether to use paragraph boundary or not.")
    
    parser.add_argument(
        '-shuf','--shuffle-type', dest='mp_shuf', type=str, required=True,
        help="Shuffling type (sentence, di, para).")
    args = parser.parse_args()
    

    main(args)
