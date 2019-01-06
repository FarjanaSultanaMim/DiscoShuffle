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
    --model-dir output/3549d721dc569f3840c6e0435a39446b
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
    --model-type nea --dropout 0.7 \
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
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 100 \
    --gradientclipnorm 5 --meanovertime \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 64 \
    --pretrained-encoder output_enc/a87b827fa7c5151192542ecb2c3af4d2
""",
"""
# TN16+PN10+pretrain(sent. shuffle, fixed)
python src/train.py \
    --fold {} \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 100 \
    --gradientclipnorm 5 --meanovertime \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 64 \
    --fix-encoder --fix-embedding \
    --pretrained-encoder output_enc/clipnorm=5.0_dropout=0.7_emb_dim=50_emb_fix=False_enc_fix=False_model_type=nea_mot=True_pretrained=False_shuf=sentence
""",
"""
# TN16+PN10+pretrain(sent. shuffle, not fixed)
python src/train.py \
    --fold {} \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 100 \
    --gradientclipnorm 5 --meanovertime \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 64 \
    --pretrained-encoder output_enc/clipnorm=5.0_dropout=0.7_emb_dim=50_emb_fix=False_enc_fix=False_model_type=nea_mot=True_pretrained=False_shuf=sentence
"""
]

f = int(sys.argv[1])

if sys.argv[2] == "train":
    comm = comm_train
    
elif sys.argv[2] == "eval":
    comm = comm_eval
    
for c in comm:
    os.system(c.format(f))
    