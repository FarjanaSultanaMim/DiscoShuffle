import sys
import os

comm_eval = ["""
# TN16
python src/eval.py \
    --fold {} \
    --model-dir output/
""",
"""
# TN16+PN10
python src/eval.py \
    --fold {} \
    --model-dir output/
""",
"""
# TN16+PN10+pretrain(di. shuffle, fixed)
python src/eval.py \
    --fold {} \
    --model-dir output/e7066f606904451be43dff3562cd06e1
""",
"""
# TN16+PN10+pretrain(di. shuffle, not fixed)
python src/eval.py \
    --fold {} \
    --model-dir output/057450e4e7e49a11c90f4fdc03e1de01
""",
"""
# TN16+PN10+pretrain(sent. shuffle, fixed)
python src/eval.py \
    --fold {} \
    --model-dir output/752aa6160e706a6ffe0f91a1e423b40a
""",
"""
# TN16+PN10+pretrain(sent. shuffle, not fixed)
python src/eval.py \
    --fold {} \
    --model-dir output/0cb7a2429b6cc73297413c20570c824f
""",
]

comm_train = ["""
# TN16
python src/train.py \
    --fold {} \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 100 \
    --gradientclipnorm 5 --meanovertime \
    --pre-trained --fix-embedding
""",
"""
# TN16+PN10
python src/train.py \
    --fold {} \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 100 \
    --gradientclipnorm 5 --meanovertime \
    --pre-trained --fix-embedding \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 64
""",
"""
# TN16+PN10+pretrain(di. shuffle, fixed)
python src/train.py \
    --fold {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 400 \
    --fix-encoder --fix-embedding \
    --pretrained-encoder output_enc/9780456c95e7c048e2501106fd40c716
""",
"""
# TN16+PN10+pretrain(di. shuffle, not fixed)
python src/train.py \
    --fold {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 400 \
    --pretrained-encoder output_enc/9780456c95e7c048e2501106fd40c716
""",
"""
# TN16+PN10+pretrain(sent. shuffle, fixed)
python src/train.py \
    --fold {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 400 \
    --fix-encoder --fix-embedding \
    --pretrained-encoder output_enc/c2c4d855a06224fd1096834eed11920d
""",
"""
# TN16+PN10+pretrain(sent. shuffle, not fixed)
python src/train.py \
    --fold {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 400 \
    --pretrained-encoder output_enc/c2c4d855a06224fd1096834eed11920d
""",
]

comm_train_enc = [
"""
# Pretraining with sentence-based shuffling
python src/train_enc.py \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --shuffle-type sentence
""",
"""
# Pretraining with DI-based shuffling
python src/train_enc.py \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --shuffle-type di
""",
]

if sys.argv[2] == "train":
    f = int(sys.argv[1])
    comm = [x.format(f) for x in comm_train]
    
elif sys.argv[2] == "eval":
    f = int(sys.argv[1])
    comm = [x.format(f) for x in comm_eval]
    
elif sys.argv[2] == "train_allfolds":
    f = int(sys.argv[1])
    comm = [comm_train[f].format(i) for i in range(0, 5)]

elif len(sys.argv) == 4 and sys.argv[3] == "train_onefold":
    f, ff = int(sys.argv[1]), int(sys.argv[2])
    comm = [comm_train[f].format(ff)]
    
elif sys.argv[2] == "eval_allfolds":
    f = int(sys.argv[1])
    comm = [comm_eval[f].format(i) for i in range(0, 5)]
    
elif sys.argv[2] == "train_enc":
    f = int(sys.argv[1])
    comm = [comm_train_enc[f]]
    
elif len(sys.argv) == 4 and sys.argv[3] == "eval_onefold":
    f, ff = int(sys.argv[1]), int(sys.argv[2])
    comm = [comm_eval[f].format(ff)]
    
        
for c in comm:
    print("===")
    print("bulkrun.py:", c)
    
    os.system(c)
    