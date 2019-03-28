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
from nltk import word_tokenize

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
    print("  # Score type: {}".format(args.mp_score_type))
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
    essayids, essays, _, scores, prompts, _ = data.load_annotated_essay_with_normalized_score('/home/mim/ICLE_essay_Wprompt.xlsx', score_source="data/{}Scores.txt".format(args.mp_score_type))
    pseqs = np.array([data.get_persing_sequence(e, p) for e, p in zip(essays, prompts)])
    
    if args.mp_di_aware:
        di_list = data.load_discourse_indicators()
        essays = data.preprocess_essay(essays, args, di_list, boseos=True)
        
    elif args.mp_model_type == "nea_aft_pretrain" and not args.mp_para:
        
        essays = data.preprocess_essay_encoder(essays, args, boseos=True)
        
    elif args.mp_no_para:
        
        essays = data.preprocess_essay_encoder(essays, args, boseos=True)
    
    else:
        essays = data.preprocess_essay(essays, args, boseos=True)
        
    if args.mp_prompt:
        
        prompts = data.preprocess_essay_encoder(prompts, args, boseos=True)
    
    # Get training and validation set!
    id2idx = dict([(v, k) for k, v in enumerate(essayids)])
    folds = data.load_folds("data/{}Folds.txt".format(args.mp_score_type), id2idx=id2idx)
    
    assert(0 <= args.fold and args.fold <= 4)
    
    tr, v, ts = data.get_fold(folds, args.fold)

    indices = np.arange(len(essays))
    main_essay_t, main_essay_v, score_t, score_v, indices_t, indices_v, prompt_t, prompt_v = essays[tr], essays[v], scores[tr], scores[v], indices[tr], indices[v], prompts[tr], prompts[v]
    pseq_t, pseq_v = pseqs[indices_t], pseqs[indices_v]
    
    print(main_essay_v[:2])
    print(prompt_v[:10])
    
    # Preparing inputs
    model_inputs_t, model_inputs_v = [], []
    
        
        
    if args.mp_elmo == True:
        
        seq_len = []
        
        for essay in main_essay_t:
            seq_len.append(len(essay))
            
        sequence_length_main = max(seq_len)
        

        main_essay_t = np.array (data.get_padded_essay(main_essay_t, sequence_length_main))
        main_essay_v = np.array(data.get_padded_essay(main_essay_v, sequence_length_main))
        
        print(main_essay_v[2])
        
        model_inputs_t += [main_essay_t]
        model_inputs_v += [main_essay_v]
        
    else:
        # Text to sequence
        if args.mp_model_type != "only_pseq":
            if args.mp_preenc != None:
                tokenizer_m = pickle.load(open(os.path.join(args.mp_preenc, "tokenizer.pickle"), "rb"))

            else:
                tokenizer_m = model.create_vocab(main_essay_t, args)

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
    
    if args.mp_pseq or args.mp_model_type == "only_pseq":
        tokenizer_pseq = model.create_vocab_seq(pseq_t, char_level=True)
        
        with open(os.path.join(out_dir, "tokenizer_pseq_f{}.pickle".format(args.fold)), "wb") as f:
            pickle.dump(tokenizer_pseq, f)
            
        sequences_train_pseq = tokenizer_pseq.texts_to_sequences(pseq_t)
        sequences_valid_pseq = tokenizer_pseq.texts_to_sequences(pseq_v)
        lens = [len(e) for e in sequences_train_pseq]

        model_inputs_t += [pad_sequences(sequences_train_pseq, maxlen=min(max(lens), data.MAX_PARAGRAPHS))]
        model_inputs_v += [pad_sequences(sequences_valid_pseq, maxlen=model_inputs_t[-1].shape[1])]

        sequence_length_pseq = model_inputs_t[-1].shape[1]
        
    #prompt
    if args.mp_prompt:
        
        tokenizer_p = model.create_vocab_prompt(prompt_t, args)
        
        with open(os.path.join(out_dir, "tokenizer_p_f{}.pickle".format(args.fold)), "wb") as f:
                pickle.dump(tokenizer_p, f)

        sequences_train_prompt = tokenizer_p.texts_to_sequences(prompt_t)
        sequences_valid_prompt = tokenizer_p.texts_to_sequences(prompt_v)
        lens = [len(e) for e in sequences_train_prompt]

        model_inputs_t += [pad_sequences(sequences_train_prompt, maxlen=min(max(lens), data.MAX_WORDS))]
        model_inputs_v += [pad_sequences(sequences_valid_prompt, maxlen=model_inputs_t[-1].shape[1])]

        sequence_length_prompt = model_inputs_t[-1].shape[1]
        
    # Create neural regression model.
    
    if args.mp_model_type == "only_pseq":
        
        mainModel = model.pseq_regression(sequence_length_pseq,
                                        args,)
        mainModel.summary()
        
    else:
        
        if args.mp_elmo == True:
            
            mainModel = model.create_regression_elmo(
                                        sequence_length_main,
                                        sequence_length_pseq,
                                        args,
                                        )
            mainModel.summary()
            
        elif args.mp_prompt:
            
            mainModel = model.create_regression_wprompt(pre_embed,
                                        tokenizer_m.word_index,
                                        tokenizer_p.word_index,
                                        sequence_length_main,
                                        sequence_length_pseq,
                                        sequence_length_prompt,
                                        args,
                                        )
            mainModel.summary()
        
        else:
            
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
     
    if args.mp_model_type == "nea_aft_pretrain":
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
    parser.add_argument(
        '-sct','--score-type', dest='mp_score_type', default="Organization",
        help="Type of score (Organization, ArgumentStrength, ThesisClarity, PromptAdherence).")
    
    # Model parameters.
    parser.add_argument(
        '-m','--model-type', dest='mp_model_type', type=str, required=True,
        help="Type of model (nea, rnn1, rnn2, nea_aft_pretrain, only_pseq).")
    parser.add_argument(
        '-d','--dropout', dest='mp_dropout', type=float, required=True,
        help="Dropout ratio.") 
    parser.add_argument(
        '-embd','--embed_elmo', dest='mp_elmo', action="store_true",
        help="Whether to use elmo embedding or not")
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
        '-ed','--embedding-dim', dest='mp_emb_dim', type=int, 
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
        '-u_lstm','--uni-lstm', dest='mp_ulstm', action="store_true",
        help="Whether to use Unidirectional LSTM or nor.")
    parser.add_argument(
        '-promt','--prompt', dest='mp_prompt', action="store_true",
        help="Whether to use prompt or not.")
    parser.add_argument(
        '-punct','--punctuation', dest='mp_punct', action="store_true",
        help="Whether to use punctuation or not.")
    parser.add_argument(
        '-para','--para', dest='mp_para', action="store_true",
        help="Whether to use punctuation or not.")
    parser.add_argument(
        '-di','--di-aware', dest='mp_di_aware', action="store_true",
        help="Discourse indicator aware model.")
    parser.add_argument(
        '-nopara','--no-para', dest='mp_no_para', action="store_true",
        help="Whether to use punctuation or not.")
    parser.add_argument(
        '-s','--seed', dest='mp_seed', type=int, 
        help="seed number.") 
    
    
    # Model parameters for essay scoring.
    parser.add_argument(
        '-pseq','--persing-seq', dest='mp_pseq', action="store_true",
        help="Use PersingNg10 sequence.")
    parser.add_argument(
        '-only-pseq','--only-persing-seq', dest='mp_only_pseq', action="store_true",
        help="Use PersingNg10 sequence.")
    parser.add_argument(
        '-pseq-embdim','--pseq-embedding-dim', dest='mp_pseq_embdim', type=int,
        help="Dimension of PersingNg10 sequence embdding.")
    parser.add_argument(
        '-pseq-encdim','--pseq-encoder-dim', dest='mp_pseq_encdim', type=int,
        help="Dimension of PersingNg10 sequence encoder.")
    parser.add_argument(
        '-pseq-conv-encdim','--pseq-conv-encoder-dim', dest='mp_pseq_conv_encdim', type=int,
        help="Dimension of PersingNg10 CONV sequence encoder.")
    
    args = parser.parse_args()
    main(args)
