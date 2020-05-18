
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

import ast


class param_t:
    def __init__(self, tp):
        ntp = []
        
        for k, v in tp:
            try:
                v = ast.literal_eval(v)
                
            except ValueError:
                pass
            except SyntaxError:
                pass
                
            ntp += [("mp_" + k, v) ]
            
        self.__dict__ = dict(ntp)
        
        if "mp_score_type" not in self.__dict__:
            self.__dict__["mp_score_type"] = "Organization" # Defaults to org.
        
        for k, v in ntp:
            self.__setattr__(k, v)

            
def main(args):
    out_dir = args.model_dir

    paramargs = param_t([ln.strip().split("=", 1) for ln in open(os.path.join(out_dir, "param.txt"), "r")])
    
    print("")
    print("Regression evaluator")
    print("  # Score type: {}".format(paramargs.mp_score_type))
    print("  # Output dir: {}".format(out_dir))
    print("  # Param string:\n{}".format(model.param_str(paramargs)))
    print("")

    # To reduce memory consumption    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Setup data 
    if paramargs.mp_score_type == "ThesisClarity": 
        essayids, essays, org_scores, scores, prompts, scaler = data.load_annotated_essay_with_normalized_score('/home/mim/ICLE_thesisClarity_Wprompt.xlsx', score_source="data/{}Scores.txt".format(paramargs.mp_score_type))
        
    else:
        
        essayids, essays, org_scores, scores, prompts, scaler = data.load_annotated_essay_with_normalized_score('/home/mim/ICLE_essay_Wprompt.xlsx', score_source="data/{}Scores.txt".format(paramargs.mp_score_type))
        
        
    pseqs = np.array([data.get_persing_sequence(e, p) for e, p in zip(essays, prompts)])
    
    if paramargs.mp_di_aware:
        di_list = data.load_discourse_indicators()
        essays = data.preprocess_essay(essays, di_list, boseos=True)
    
    elif paramargs.mp_model_type == "nea_aft_pretrain" and not paramargs.mp_para:
        
        essays = data.preprocess_essay_encoder(essays, paramargs, boseos=True)
        
    elif paramargs.mp_no_para:
        
        essays = data.preprocess_essay_encoder(essays, paramargs, boseos=True)
    
    else:
        essays = data.preprocess_essay(essays, paramargs, boseos=True)
        
    if paramargs.mp_prompt:
        
        prompts = data.preprocess_essay_encoder(prompts, paramargs, boseos=True)
    
    # Get training and validation set!
    id2idx = dict([(v, k) for k, v in enumerate(essayids)])
    folds = data.load_folds("data/{}Folds.txt".format(paramargs.mp_score_type), id2idx=id2idx)
    
    assert(0 <= args.fold and args.fold <= 4)
    
    _, _, ts = data.get_fold(folds, args.fold)

    indices = np.arange(len(essays))
    main_essay_t, main_essay_v, score_t, score_v, indices_t, indices_v, prompt_t, prompt_v = [], essays[ts], [], org_scores[ts], [], indices[ts], [], prompts[ts]
    pseq_t, pseq_v = [], pseqs[indices_v]
    
    print(main_essay_v[:1])
    print(prompt_v[:1])
    
    # Preparing inputs
    model_inputs_v = []
    
    # Text to sequence
    if paramargs.mp_model_type != "only_pseq":
        
        tokenizer_m = pickle.load(open(os.path.join(args.model_dir, "tokenizer_f{}.pickle".format(args.fold)), "rb"))

        sequences_valid_main = tokenizer_m.texts_to_sequences(main_essay_v)
        lens = [len(e) for e in sequences_valid_main]

        model_inputs_v += [pad_sequences(sequences_valid_main, maxlen=min(max(lens), data.MAX_WORDS))]

        sequence_length_main = model_inputs_v[-1].shape[1]
    
    # Persing sequence to sequence
    sequence_length_pseq = None
    
    if paramargs.mp_pseq or paramargs.mp_model_type == "only_pseq":
        
        tokenizer_pseq = pickle.load(open(os.path.join(args.model_dir, "tokenizer_pseq_f{}.pickle".format(args.fold)), "rb"))
        sequences_valid_pseq = tokenizer_pseq.texts_to_sequences(pseq_v)
        lens = [len(e) for e in sequences_valid_pseq]

        model_inputs_v += [pad_sequences(sequences_valid_pseq, maxlen=min(max(lens), data.MAX_PARAGRAPHS))]

        sequence_length_pseq = model_inputs_v[-1].shape[1]
        
      #prompt
    if paramargs.mp_prompt:
        
        tokenizer_p = pickle.load(open(os.path.join(args.model_dir, "tokenizer_p_f{}.pickle".format(args.fold)), "rb"))
        
        sequences_valid_prompt = tokenizer_p.texts_to_sequences(prompt_v)
        lens = [len(e) for e in sequences_valid_prompt]

        model_inputs_v += [pad_sequences(sequences_valid_prompt, maxlen=min(max(lens), data.MAX_WORDS))]

        sequence_length_prompt = model_inputs_v[-1].shape[1]

    
    if paramargs.mp_model_type == "only_pseq":
        
        mainModel = model.pseq_regression(sequence_length_pseq,
                                        paramargs,)
        mainModel.summary()    
    
    else:
        
        if paramargs.mp_prompt:
            
            mainModel = model.create_regression_wprompt(None,
                                        tokenizer_m.word_index,
                                        tokenizer_p.word_index,
                                        sequence_length_main,
                                        sequence_length_pseq,
                                        sequence_length_prompt,
                                        paramargs,
                                        )
            mainModel.summary()
        
        else:
            
            mainModel = model.create_regression(None,
                                        tokenizer_m.word_index,
                                        sequence_length_main,
                                        sequence_length_pseq,
                                        paramargs,
                                        )
            mainModel.summary()
    
    mainModel.load_weights(os.path.join(args.model_dir, "regression_f{}.hdf5".format(args.fold)), by_name=True)

    print("Starting evaluation.")
    
    # Perform prediction!
    score_model = mainModel.predict(
        model_inputs_v,
        verbose=1,
        batch_size=32)
    
    
    score_model_n = scaler.inverse_transform(score_model)
    score_model_n = score_model_n.tolist()
    
    score_model_nf = []
    for i in score_model_n:
        for j in i:
            score_model_nf.append(j)
    
    # Save to the file.
    with open(os.path.join(out_dir, "prediction_f{}.json".format(args.fold)), "w") as f:
        mse, mae = mean_squared_error(score_v, score_model_nf), mean_absolute_error(score_v, score_model_nf)
        
        pr = {
            "system": score_model_nf,
            "gold": score_v.tolist(),
            "MSE": mse,
            "MAE": mae,
        }
        
        json.dump(pr, f)
    
        print("MSE:", mse)
        print("MAE:", mae)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-fo','--fold', dest='fold', type=int, required=True,
        help="Fold ID ([1, 5]).")
    parser.add_argument(
        '-m','--model-dir', dest='model_dir', type=str, required=True,
        help="Path to model.")
    parser.add_argument(
        '-punct','--punctuation', dest='mp_punct', action="store_true",
        help="Whether to use punctuation or not.")

    args = parser.parse_args()
    main(args)
