import sys
import os

comm_train = [

"""
# 1: Base+PFE
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 200 \
    --gradientclipnorm 0 --meanovertime \
    --pre-trained --fix-embedding --punctuation --para \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 200
""",
              
 """
# 2: Base+PE
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 200 --prompt \
    --gradientclipnorm 0 --meanovertime \
    --pre-trained --fix-embedding --punctuation --para \
""",
                
              
                          
              
"""
# 3: Base+PFE+pretrained(sentence-shuffled, fine-tuned)
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 200 \
    --gradientclipnorm 0 --meanovertime --punctuation \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 200 \
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",
              """
# 4: Base+PFE+pretrained(sentence-shuffled, fixed-encoder)
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 200 \
    --gradientclipnorm 0 --meanovertime --punctuation \
    --fix-encoder --fix-embedding \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 200 \
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",
              
              """
# 5: Base+PE+pretrained(sentence-shuffle, fine-tuned)
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 200 \
    --gradientclipnorm 0 --meanovertime --punctuation --prompt \
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",
"""
# 6: Base+PE+pretrained(sentence-shuffle, fixed-encoder)
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 200 \
    --fix-encoder --fix-embedding \
    --gradientclipnorm 0 --meanovertime --punctuation --prompt \
    --pretrained-encoder PATH_TO_PRETRAINED_ENCODER
""",
              
                            

]






comm_train_enc = [
"""
# 1: Pre-Pretraining with sentence-based shuffling
python src/train_enc.py \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 200 \
    --gradientclipnorm 5 --meanovertime --pre-trained \
    --shuffle-type sentence --punctuation --essay-selection  AllEssay
""",    
"""
# 2: Pretraining with sentence-based shuffling
python src/train_enc.py \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 200 \
    --gradientclipnorm 5 --meanovertime \
    --shuffle-type sentence --punctuation --essay-selection icle\
    --pretrained-encoder PATH_TO_OUTPUT_ENCODER
""",
    
    
    
"""
# 3: Pre-Pretraining with DI-based shuffling
python src/train_enc.py \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 200 \
    --gradientclipnorm 5 --meanovertime --pre-trained \
    --shuffle-type di --punctuation --essay-selection  AllEssay
""",
"""
# 4: Pretraining with DI-based shuffling
python src/train_enc.py \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 200 \
    --gradientclipnorm 5 --meanovertime \
    --shuffle-type di --punctuation --essay-selection icle\
    --pretrained-encoder PATH_TO_OUTPUT_ENCODER
""", 
    
 
"""
# 5: Pre-Pretraining with paragraph-based shuffling
python src/train_enc.py \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 200 \
    --gradientclipnorm 5 --meanovertime --pre-trained \
    --shuffle-type para --punctuation --w-para  --essay-selection ICLEandTOEFL11
""",
"""
# 6: Pretraining with paragraph-based shuffling
python src/train_enc.py \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 200 \
    --gradientclipnorm 5 --meanovertime  \
    --shuffle-type para --punctuation --w-para --essay-selection icle\
    --pretrained-encoder PATH_TO_OUTPUT_ENCODER
""",

]
    
    
    
 



comm_eval = ["""
# eval
python src/eval.py \
    --fold {} \
    --model-dir output/
""",    
]

comm_eval_homo = """
# NEA
python src/eval.py \
    --fold {} \
    --model-dir {}
"""


if sys.argv[1] == "train":
    sct = sys.argv[2]
    f = int(sys.argv[3])
    comm = [x.format(f, sct) for x in comm_train]
    
elif sys.argv[1] == "train_allfolds":
    sct = sys.argv[2]
    f = int(sys.argv[3])
    comm = [comm_train[f].format(i, sct) for i in range(0, 5)]

elif sys.argv[1] == "train_onefold_for_hptune":
    sct = sys.argv[2]
    f = int(sys.argv[3])
    ff = int(sys.argv[4])
    comm = []
    
    for dropout in [0.5, 0.7, 0.9]:
        for gcn in [0, 3, 5, 7]:
            cmd = comm_train[f]
            cmd = cmd.replace("--dropout 0.7", "--dropout {}".format(dropout))
            cmd = cmd.replace("--gradientclipnorm 0", "--gradientclipnorm {}".format(gcn))
            comm += [cmd.format(ff, sct)]
            
elif sys.argv[1] == "train_allfolds_for_hptune":
    sct = sys.argv[2]
    f = int(sys.argv[3])
    comm = []
    for i in range(0, 5):
        for dropout in [0.5, 0.7, 0.9]:
            for gcn in [0, 3, 5, 7]:
                cmd = comm_train[f]
                cmd = cmd.replace("--dropout 0.7", "--dropout {}".format(dropout))
                cmd = cmd.replace("--gradientclipnorm 0", "--gradientclipnorm {}".format(gcn))
                comm += [cmd.format(i, sct)]

elif sys.argv[1] == "train_allfolds_for_hptune_wSeed":
    sct = sys.argv[2]
    f = int(sys.argv[3])
    comm = []
    for seed in [0, 1, 2, 3, 4]:
        for i in range(0, 5):
            for dropout in [0.5, 0.7, 0.9]:
                for gcn in [0, 3, 5, 7]:
                    cmd = comm_train[f]
                    cmd = cmd.replace("--seed 0", "--seed {}".format(seed))
                    cmd = cmd.replace("--dropout 0.7", "--dropout {}".format(dropout))
                    cmd = cmd.replace("--gradientclipnorm 0", "--gradientclipnorm {}".format(gcn))
                    comm += [cmd.format(i, sct)]
                
elif len(sys.argv) == 5 and sys.argv[1] == "train_onefold":
    sct = sys.argv[2]
    f, ff = int(sys.argv[3]), int(sys.argv[4])
    comm = [comm_train[f].format(ff, sct)]
    
elif sys.argv[1] == "eval":
    f = int(sys.argv[2])
    comm = [x.format(f) for x in comm_eval]
    
elif sys.argv[1] == "eval_allfolds":
    f = int(sys.argv[2])
    comm = [comm_eval[f].format(i) for i in range(0, 5)]
    
elif sys.argv[1] == "eval_allfolds_homo":
    model_dir = sys.argv[2]
    comm = [comm_eval_homo.format(i, model_dir) for i in range(0, 5)]
    
elif len(sys.argv) == 4 and sys.argv[1] == "eval_onefold":
    f, ff = int(sys.argv[2]), int(sys.argv[3])
    comm = [comm_eval[f].format(ff)]
    
elif sys.argv[1] == "eval_onefold_homo":
    model_dir = sys.argv[2]
    ff = int(sys.argv[3])
    comm = [comm_eval_homo.format(ff, model_dir)]
    
elif sys.argv[1] == "train_enc":
    f = int(sys.argv[2])
    comm = [comm_train_enc[f]]
    
elif sys.argv[1] == "eval_allfolds_for_hptune":
    f = int(sys.argv[2])
    comm = []  
    folder = [  'folder_name_1, folder_name_2, folder_name_3, folder_name_4, folder_name_5'
             ]
    
    for i in range(0, 5):
        cmd = comm_eval[f]
        if i==0:
            cmd = cmd.replace("--model-dir output/", "--model-dir {}".format(folder[0]))
        if i==1:
            cmd = cmd.replace("--model-dir output/", "--model-dir {}".format(folder[1]))
        if i==2:
            cmd = cmd.replace("--model-dir output/", "--model-dir {}".format(folder[2]))
        if i==3:
            cmd = cmd.replace("--model-dir output/", "--model-dir {}".format(folder[3]))
        if i==4:
            cmd = cmd.replace("--model-dir output/", "--model-dir {}".format(folder[4]))
        comm += [cmd.format(i)]
    
        
for c in comm:
    print("===")
    print("bulkrun.py:", c)
    
    os.system(c)
    