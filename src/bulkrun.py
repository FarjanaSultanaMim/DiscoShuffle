import sys
import os

comm_eval = ["""
# TN16
python src/eval.py \
    --fold {} \
    --model-dir output/988c82dd83f6a34df8005ac89db85465
""",
"""
# TN16+PN10
python src/eval.py \
    --fold {} \
    --model-dir output/2966a197dd56de18a9973720c9219a0d
""",
"""
# TN16+PN10+pretrain(di. shuffle, fixed)
python src/eval.py \
    --fold {} \
    --model-dir output/eb40209496004f29ca30acb28e49c665
""",
"""
# TN16+PN10+pretrain(di. shuffle, not fixed)
python src/eval.py \
    --fold {} \
    --model-dir output/e0ab2e7b43b37719666bb6e1b0e58e10
""",
"""
# TN16+PN10+pretrain(sent. shuffle, fixed)
python src/eval.py \
    --fold {} \
    --model-dir output/fa37bf66f2563eec16382c1eac16a108
""",
"""
# TN16+PN10+pretrain(sent. shuffle, not fixed)
python src/eval.py \
    --fold {} \
    --model-dir output/45209d4279bbad14c42f65bbc19aa2f3
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
    --embedding-dim 50 --aggregation-grudim 100 \
    --gradientclipnorm 5 --meanovertime \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 64 \
    --fix-encoder --fix-embedding \
    --pretrained-encoder output_enc/a87b827fa7c5151192542ecb2c3af4d2
""",
"""
# TN16+PN10+pretrain(di. shuffle, not fixed)
python src/train.py \
    --fold {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 100 \
    --gradientclipnorm 5 --meanovertime \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 64 \
    --pretrained-encoder output_enc/a87b827fa7c5151192542ecb2c3af4d2
""",
"""
# TN16+PN10+pretrain(sent. shuffle, fixed)
python src/train.py \
    --fold {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 100 \
    --gradientclipnorm 5 --meanovertime \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 64 \
    --fix-encoder --fix-embedding \
    --pretrained-encoder output_enc/750570aed2d16633ecbe4237d2d95b71
""",
"""
# TN16+PN10+pretrain(sent. shuffle, not fixed)
python src/train.py \
    --fold {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 100 \
    --gradientclipnorm 5 --meanovertime \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 64 \
    --pretrained-encoder output_enc/750570aed2d16633ecbe4237d2d95b71
"""
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

elif sys.argv[2] == "eval_allfolds":
    f = int(sys.argv[1])
    comm = [comm_eval[f].format(i) for i in range(0, 5)]
        
for c in comm:
    print("===")
    print("bulkrun.py:", c)
    
    os.system(c)
    