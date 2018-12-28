
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
        
        for k, v in ntp:
            self.__setattr__(k, v)

            
def main(args):
    out_dir = args.model_dir

    paramargs = param_t([ln.strip().split("=", 1) for ln in open(os.path.join(out_dir, "param.txt"), "r")])
    
    print("")
    print("Regression evaluator")
    print("  # Output dir: {}".format(out_dir))
    print("  # Param string:\n{}".format(model.param_str(paramargs)))
    print("")

    # To reduce memory consumption    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Setup data  
    essayids, essays, scores, prompts = data.load_annotated_essay("/home/mim/ICLE_essay_Wprompt.xlsx")
    pseqs = np.array([data.get_persing_sequence(e, p) for e, p in zip(essays, prompts)])
    
    if paramargs.mp_di_aware:
        di_list = data.load_discourse_indicators()
        essays = data.preprocess_essay(essays, di_list, boseos=True)
        
    else:
        essays = data.preprocess_essay(essays, boseos=True)
    
    # Get training and validation set!
    id2idx = dict([(v, k) for k, v in enumerate(essayids)])
    folds = data.load_folds(id2idx=id2idx)
    
    assert(0 <= args.fold and args.fold <= 4)
    
    _, _, ts = data.get_fold(folds, args.fold)

    indices = np.arange(len(essays))
    main_essay_t, main_essay_v, score_t, score_v, indices_t, indices_v = [], essays[ts], [], scores[ts], [], indices[ts]
    pseq_t, pseq_v = [], pseqs[indices_v]
    
    # Preparing inputs
    model_inputs_v = []
    
    # Text to sequence
    tokenizer_m = pickle.load(open(os.path.join(args.model_dir, "tokenizer_f{}.pickle".format(args.fold)), "rb"))

    sequences_valid_main = tokenizer_m.texts_to_sequences(main_essay_v)
    lens = [len(e) for e in sequences_valid_main]

    model_inputs_v += [pad_sequences(sequences_valid_main, maxlen=min(max(lens), data.MAX_WORDS))]

    sequence_length_main = model_inputs_v[-1].shape[1]
    
    # Persing sequence to sequence
    sequence_length_pseq = None
    
    if paramargs.mp_pseq:
        tokenizer_pseq = pickle.load(open(os.path.join(args.model_dir, "tokenizer_pseq_f{}.pickle".format(args.fold)), "rb"))
        sequences_valid_pseq = tokenizer_pseq.texts_to_sequences(pseq_v)
        lens = [len(e) for e in sequences_valid_pseq]

        model_inputs_v += [pad_sequences(sequences_valid_pseq, maxlen=min(max(lens), data.MAX_PARAGRAPHS))]

        sequence_length_pseq = model_inputs_v[-1].shape[1]
        
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
    
    # Save to the file.
    with open(os.path.join(out_dir, "prediction_f{}.json".format(args.fold)), "w") as f:
        mse, mae = mean_squared_error(score_v, score_model), mean_absolute_error(score_v, score_model)
        
        pr = {
            "system": score_model.tolist(),
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

    args = parser.parse_args()
    main(args)
