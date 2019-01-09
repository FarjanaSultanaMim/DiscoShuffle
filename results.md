BEST:

Earlystopping = 15, Bidirectional_GRU

# Baselines
## TN16

Best hyperparameter based on fold no. 2 and 4 among 5 folds:

fold_1:

clipnorm 0, dropout 0.3 = val_loss: 0.0259
clipnorm 0, dropout 0.5 = val_loss: 0.0258
clipnorm 0, dropout 0.7 = val_loss: 0.0256
clipnorm 0, dropout 0.8 = val_loss: 0.0256
clipnorm 0, dropout 0.9 = val_loss: 0.0256

clipnorm 1, dropout 0.7 = val_loss: 0.0257
clipnorm 1, dropout 0.5 = val_loss: 0.0257

clipnorm 5, dropout 0.7 = val_loss: 0.0257
clipnorm 5, dropout 0.5 = val_loss: 0.0257
clipnorm 5, dropout 0.9 = val_loss: 0.0256

clipnorm 3, dropout 0.5 = val_loss: 0.0257
clipnorm 3, dropout 0.7 = val_loss: 0.0256
clipnorm 3, dropout 0.9 = val_loss: 0.0256

fold_3:

clipnorm 0, dropout 0.7 = val_loss: 0.0493
clipnorm 0, dropout 0.9 = val_loss: 0.0505

clipnorm 5, dropout 0.9 = val_loss: 0.0493
clipnorm 5, dropout 0.7 = val_loss: 0.0493

clipnorm 3, dropout 0.7 = val_loss: 0.0490
clipnorm 3, dropout 0.9 = val_loss: 0.0541

```
CUDA_VISIBLE_DEVICES=0 python src/train.py  --fold 0   --model-type nea --dropout 0.7    --embedding-dim 50 --aggregation-grudim 300  --gradientclipnorm 3 --meanovertime   --pre-trained --fix-embedding
```

fold_0: MSE: 0.3621767404762582, MAE: 0.43679335951805115
fold_1: MSE: 0.3298523836700365, MAE: 0.41286392235637304
fold_2: MSE: 0.3931328914678891, MAE: 0.42098302983525976
fold_3: MSE: 0.30799082403575084,MAE: 0.42356242943758987
fold_4: MSE: 0.36748199519640024,MAE: 0.46359519481658934

- MSE: 0.352127


## TN16+PN10

Best hyperparameter based on fold no. 2 and 4 among 5 folds:

fold_1:

clipnorm 0, dropout 0.3 = val_loss: 0.0220
clipnorm 0, dropout 0.5 = val_loss: 0.0216
clipnorm 0, dropout 0.7 = val_loss: 0.0212
clipnorm 0, dropout 0.9 = val_loss: 0.0255


clipnorm 5, dropout 0.5 = val_loss: 0.0226
clipnorm 5, dropout 0.7 = val_loss: 0.0221
clipnorm 5, dropout 0.9 = val_loss: 0.0208

clipnorm 3, dropout 0.5 = val_loss: 0.0229 
clipnorm 3, dropout 0.7 = val_loss: 0.0215
clipnorm 3, dropout 0.9 = val_loss: 0.0233

fold_3:

clipnorm 0, dropout 0.7 = val_loss: 0.0203
clipnorm 5, dropout 0.9 = val_loss: 0.0181
clipnorm 3, dropout 0.7 = val_loss: 0.0205

```
CUDA_VISIBLE_DEVICES=0 python src/train.py     --fold 0     --model-type nea --dropout 0.9     --embedding-dim 50 --aggregation-grudim 300     --gradientclipnorm 5 --meanovertime     --pre-trained --fix-embedding     --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 400
```

fold_0: MSE: 0.18817686358418173, MAE: 0.35129995703697203
fold_1: MSE: 0.18391588820501945, MAE: 0.3496616467907654
fold_2: MSE: 0.1780143176682396, MAE: 0.35123343906592375
fold_3: MSE: 0.20512577761926526, MAE: 0.3599928285352033
fold_4: MSE: 0.2213006920121294, MAE: 0.3716801816225052

- MSE: 0.1953067


# Document encoder pretraining

## With DI shuffling

### Pretraining

```
{'aggr_grudim': '300', 'att': 'False', 'clipnorm': '5.0', 'dropout': '0.7', 'emb_dim': '50', 'emb_fix': 'False', 'enc_fix': 'False', 'encdim': 'None', 'model_type': 'nea', 'mot': 'True', 'pretrained': 'False', 'shuf': 'di'}
```

val_acc: 0.879913


### TN16+PN10+pretrain (di. shuffle, no fine-tuning)

- MSEs: 
- MSE:


### TN16+PN10+pretrain (di. shuffle, with fine-tuning)

- MSEs: 
- MSE: 


## With sentence shuffling

### Pretraining

```
aggr_grudim=300
att=False
clipnorm=5.0
dropout=0.7
emb_dim=50
emb_fix=False
enc_fix=False
encdim=None
model_type=nea
mot=True
pretrained=False
shuf=sentence
```

val_acc: 0.732533


### TN16+PN10+pretrain (sent. shuffle, no fine-tuning)

```
aggr_grudim=300
att=False
clipnorm=5.0
di_aware=False
dropout=0.7
emb_dim=50
emb_fix=True
enc_fix=True
encdim=None
model_type=nea_aft_pretrain
mot=True
preenc=output_enc/c2c4d855a06224fd1096834eed11920d
pretrained=False
pseq=True
pseq_embdim=16
pseq_encdim=400
```

- MSEs: [0.1708993219855502, 0.188276010211909, 0.16168191047724936, 0.21673573230273174, 0.21641181262379064]
- MSE: (0.1908009575202462, STDEV: 0.022710780694247448)	


### TN16+PN10+pretrain (sent. shuffle, with fine-tuning)

```
aggr_grudim=300
att=False
clipnorm=5.0
di_aware=False
dropout=0.7
emb_dim=50
emb_fix=False
enc_fix=False
encdim=None
model_type=nea_aft_pretrain
mot=True
preenc=output_enc/c2c4d855a06224fd1096834eed11920d
pretrained=False
pseq=True
pseq_embdim=16
pseq_encdim=400
```

- MSEs: [0.17768359446192292, 0.16614793271124353, 0.1609080225191747, 0.20891687420427643, 0.21897310227394462]
- MSE: 0.18652590523411244, STDEV: 0.023254677920348094

