import argparse, logging

import pickle
import json

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
    out_dir = os.path.join("output", out_dir)
    os.system("mkdir -p {}".format(out_dir))
    
    with open(os.path.join(out_dir, "param.txt"), "w") as f:
        print(pstr, file=f)

    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f)
        
    print("")
    print("Regression trainer")
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
    data.get_normalized_score_and_save('/home/mim/ICLE_essay_Wprompt.xlsx')
    essayids, essays, scores, prompts = data.load_essay_with_normalized_score("normalized_df.csv")
    pseqs = np.array([data.get_persing_sequence(e, p) for e, p in zip(essays, prompts)])
    
    if args.mp_di_aware:
        di_list = data.load_discourse_indicators()
        essays = data.preprocess_essay(essays, di_list, boseos=True)
        
    else:
        essays = data.preprocess_essay(essays, boseos=True)
    
    # Get training and validation set!
    id2idx = dict([(v, k) for k, v in enumerate(essayids)])
    folds = data.load_folds(id2idx=id2idx)
    
    assert(0 <= args.fold and args.fold <= 4)
    
    tr, v, ts = data.get_fold(folds, args.fold)

    indices = np.arange(len(essays))
    main_essay_t, main_essay_v, score_t, score_v, indices_t, indices_v = essays[tr], essays[v], scores[tr], scores[v], indices[tr], indices[v]
    pseq_t, pseq_v = pseqs[indices_t], pseqs[indices_v]
    
    # Preparing inputs
    model_inputs_t, model_inputs_v = [], []
    
    # Text to sequence
    if args.mp_preenc != None:
        tokenizer_m = pickle.load(open(os.path.join(args.mp_preenc, "tokenizer.pickle"), "rb"))
        
    else:
        tokenizer_m = model.create_vocab(main_essay_t)
        
    with open(os.path.join(out_dir, "tokenizer_f{}.pickle".format(args.fold)), "wb") as f:
        pickle.dump(tokenizer_m, f)
    
    sequences_train_main = tokenizer_m.texts_to_sequences(main_essay_t)
    sequences_valid_main = tokenizer_m.texts_to_sequences(main_essay_v)
    lens = [len(e) for e in sequences_train_main]

    model_inputs_t += [pad_sequences(sequences_train_main, maxlen=min(max(lens), data.MAX_WORDS))]
    model_inputs_v += [pad_sequences(sequences_valid_main, maxlen=model_inputs_t[-1].shape[1])]

    sequence_length_main = model_inputs_t[-1].shape[1]
    
    # Persing sequence to sequence
    sequence_length_pseq = None
    
    if args.mp_pseq:
        tokenizer_pseq = model.create_vocab_seq(pseq_t, char_level=True)
        
        with open(os.path.join(out_dir, "tokenizer_pseq_f{}.pickle".format(args.fold)), "wb") as f:
            pickle.dump(tokenizer_pseq, f)
            
        sequences_train_pseq = tokenizer_pseq.texts_to_sequences(pseq_t)
        sequences_valid_pseq = tokenizer_pseq.texts_to_sequences(pseq_v)
        lens = [len(e) for e in sequences_train_pseq]

        model_inputs_t += [pad_sequences(sequences_train_pseq, maxlen=min(max(lens), data.MAX_PARAGRAPHS))]
        model_inputs_v += [pad_sequences(sequences_valid_pseq, maxlen=model_inputs_t[-1].shape[1])]

        sequence_length_pseq = model_inputs_t[-1].shape[1]
        
    # Create neural regression model.
    mainModel = model.create_regression(pre_embed,
                                        tokenizer_m.word_index,
                                        sequence_length_main,
                                        sequence_length_pseq,
                                        args,
                                        )
    mainModel.summary()
    
    if args.mp_preenc != None:
        mainModel.load_weights(os.path.join(args.mp_preenc, "encoder.hdf5"), by_name=True)

    print("Starting training.")
    
    optimizer_main=keras.optimizers.Adam(clipnorm=args.mp_clipnorm)
    mainModel.compile(optimizer=optimizer_main, loss='mse', metrics=['mse'])
    
    
    es=keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=15,
                                      verbose=0, mode='auto', baseline=None,
                                      restore_best_weights=True)

    nbl = kr_util.NBatchLogger(os.path.join(out_dir, "logs_f{}.pickle".format(args.fold)))

    mainModel.fit(
        model_inputs_t,
        score_t,
        validation_data=(model_inputs_v, score_v),
        epochs=100, verbose=1, callbacks=[es, nbl],
        batch_size=32)

    mainModel.save_weights(os.path.join(out_dir, "regression_f{}.hdf5".format(args.fold)))
    
    print()
    print("# Output dir: {}".format(out_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-fo','--fold', dest='fold', type=int, required=True,
        help="Fold ID ([1, 5]).")
    
    # Model parameters.
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
        '-enc','--pretrained-encoder', dest='mp_preenc', type=str,
        help="Path to pretrained encoder.")
    parser.add_argument(
        '-di','--di-aware', dest='mp_di_aware', action="store_true",
        help="Discourse indicator aware model.")
    
    # Model parameters for essay scoring.
    parser.add_argument(
        '-pseq','--persing-seq', dest='mp_pseq', action="store_true",
        help="Use PersingNg10 sequence.")
    parser.add_argument(
        '-pseq-embdim','--pseq-embedding-dim', dest='mp_pseq_embdim', type=int,
        help="Dimension of PersingNg10 sequence embdding.")
    parser.add_argument(
        '-pseq-encdim','--pseq-encoder-dim', dest='mp_pseq_encdim', type=int,
        help="Dimension of PersingNg10 sequence encoder.")
    
    args = parser.parse_args()
    main(args)
