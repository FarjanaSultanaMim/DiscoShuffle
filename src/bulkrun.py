import sys
import os

comm_eval_homo = """
# TN16
python src/eval.py \
    --fold {} \
    --model-dir {}
"""
             
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
    --model-dir output/46f970b1c6379f85b7ccc1fe68a8af14
""",
"""
# TN16+PN10+pretrain(di. shuffle, not fixed)
python src/eval.py \
    --fold {} \
    --model-dir output/0aa2570ced889c4e88ae1554253cb412
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
"""
# TN16+PN10+pretrain(sent. shuffle, fixed, no pseq)
python src/eval.py \
    --fold {} \
    --model-dir output/f8a2a3674079b877485038b9f59818ff
""",
"""
# TN16+PN10+pretrain(sent. shuffle, not fixed, no pseq)
python src/eval.py \
    --fold {} \
    --model-dir output/a2fe2d615652d3170309fc000cd559be
""",             
"""
# TN16+PN10+pretrain(di. shuffle, fixed, no pseq)
python src/eval.py \
    --fold {} \
    --model-dir output/9ae037a97e5283d4651b77444b97dd42
""",
"""
# TN16+PN10+pretrain(di. shuffle, not fixed, no pseq)
python src/eval.py \
    --fold {} \
    --model-dir output/b65364664e64110a43ca71623a066182
""",             
]

comm_train = ["""
# TN16
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea --dropout 0.25 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --pre-trained --fix-embedding
""",
"""
# TN16+PN10
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --pre-trained --fix-embedding \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 400
""",
"""
# TN16+PN10+pretrain(di. shuffle, fixed)
python src/train.py \
    --fold {} \
    --score-type {} \
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
    --score-type {} \
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
    --score-type {} \
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
    --score-type {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 400 \
    --pretrained-encoder output_enc/c2c4d855a06224fd1096834eed11920d
""",
"""
# 6: TN16+pretrain(sent. shuffle, fixed)
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --fix-encoder --fix-embedding \
    --pretrained-encoder output_enc/c2c4d855a06224fd1096834eed11920d
""",
"""
# 7: TN16+pretrain(sent. shuffle, not fixed)
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --pretrained-encoder output_enc/c2c4d855a06224fd1096834eed11920d
""",
"""
# 8: TN16+pretrain(di. shuffle, fixed)
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --fix-encoder --fix-embedding \
    --pretrained-encoder output_enc/9780456c95e7c048e2501106fd40c716
""",
"""
# 9: TN16+pretrain(di. shuffle, not fixed)
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea_aft_pretrain --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --pretrained-encoder output_enc/9780456c95e7c048e2501106fd40c716
""",
"""
# 10: TN16 (random encoder)
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --fix-encoder --fix-embedding
""",
"""
# 11: TN16+PN10 (random encoder)
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea --dropout 0.7 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --fix-encoder --fix-embedding \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 400
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
    
    for dropout in [0.25, 0.5, 0.75]:
        for gcn in [5, 10]:
            cmd = comm_train[f]
            cmd = cmd.replace("--dropout 0.7", "--dropout {}".format(dropout))
            cmd = cmd.replace("--gradientclipnorm 5", "--gradientclipnorm {}".format(gcn))
            comm += [cmd.format(ff, sct)]
    
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
    
        
for c in comm:
    print("===")
    print("bulkrun.py:", c)
    
    os.system(c)
    