BEST:

Earlystopping = 15, Bidirectional_GRU

# Baselines
## TN16

Best hyperparameter based on fold no. 2 and 4 among 5 folds:


### fold_1:

- clipnorm 0, dropout 0.3 = val_loss: 0.0257
- clipnorm 0, dropout 0.5 = val_loss: 0.0258
- clipnorm 0, dropout 0.7 = val_loss: 0.0256
- clipnorm 0, dropout 0.8 = val_loss: 0.0256
- clipnorm 0, dropout 0.9 = val_loss: 0.0256

- clipnorm 1, dropout 0.7 = val_loss: 0.0257
- clipnorm 1, dropout 0.5 = val_loss: 0.0257

- clipnorm 5, dropout 0.7 = val_loss: 0.0257
- clipnorm 5, dropout 0.5 = val_loss: 0.0257
- clipnorm 5, dropout 0.9 = val_loss: 0.0256

- clipnorm 3, dropout 0.5 = val_loss: 0.0257
- clipnorm 3, dropout 0.7 = val_loss: 0.0256
- clipnorm 3, dropout 0.9 = val_loss: 0.0256


### fold_3:

- clipnorm 0, dropout 0.7 = val_loss: 0.0493
- clipnorm 0, dropout 0.9 = val_loss: 0.0505

- clipnorm 5, dropout 0.9 = val_loss: 0.0493
- clipnorm 5, dropout 0.7 = val_loss: 0.0493

- clipnorm 3, dropout 0.7 = val_loss: 0.0490
- clipnorm 3, dropout 0.9 = val_loss: 0.0541


### Final results

```
CUDA_VISIBLE_DEVICES=0 python src/train.py  --fold 0   --model-type nea --dropout 0.7    --embedding-dim 50 --aggregation-grudim 300  --gradientclipnorm 3 --meanovertime   --pre-trained --fix-embedding
```

- fold_0: MSE: 0.3621767404762582, MAE: 0.43679335951805115
- fold_1: MSE: 0.3298523836700365, MAE: 0.41286392235637304
- fold_2: MSE: 0.3931328914678891, MAE: 0.42098302983525976
- fold_3: MSE: 0.30799082403575084,MAE: 0.42356242943758987
- fold_4: MSE: 0.36748199519640024,MAE: 0.46359519481658934

- MSE: 0.352127


## TN16+PN10: Organization

Best hyperparameter based on fold no. 2 and 4 among 5 folds:


### fold_1:

- clipnorm 0, dropout 0.3 = val_loss: 0.0220
- clipnorm 0, dropout 0.5 = val_loss: 0.0216
- clipnorm 0, dropout 0.7 = val_loss: 0.0212
- clipnorm 0, dropout 0.9 = val_loss: 0.0255


- clipnorm 5, dropout 0.5 = val_loss: 0.0226
- clipnorm 5, dropout 0.7 = val_loss: 0.0221
- clipnorm 5, dropout 0.9 = val_loss: 0.0208

- clipnorm 3, dropout 0.5 = val_loss: 0.0229 
- clipnorm 3, dropout 0.7 = val_loss: 0.0215
- clipnorm 3, dropout 0.9 = val_loss: 0.0233


### fold_3:

- clipnorm 0, dropout 0.7 = val_loss: 0.0203
- clipnorm 5, dropout 0.9 = val_loss: 0.0181
- clipnorm 3, dropout 0.7 = val_loss: 0.0205


### Final results

```
CUDA_VISIBLE_DEVICES=0 python src/train.py     --fold 0     --model-type nea --dropout 0.9     --embedding-dim 50 --aggregation-grudim 300     --gradientclipnorm 5 --meanovertime     --pre-trained --fix-embedding     --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 400
```
dir: f3266eb6be217d61a69d21977169665f

- fold_0: MSE: 0.18253937013255694, MAE: 0.34780204176902774
- fold_1: MSE: 0.1700185757241532, MAE: 0.3373449441805408
- fold_2: MSE: 0.17473423035119662, MAE: 0.3415010995532743
- fold_3: MSE: 0.21384976284436022, MAE: 0.3624011325598949
- fold_4: 0.23140004653422813, MAE: 0.3745786982774735

- MSE: 0.1945084


# Document encoder pretraining

## With DI shuffling

### Pretraining

```
{'aggr_grudim': '300', 'att': 'False', 'clipnorm': '5.0', 'dropout': '0.7', 'emb_dim': '50', 'emb_fix': 'False', 'enc_fix': 'False', 'encdim': 'None', 'model_type': 'nea', 'mot': 'True', 'pretrained': 'False', 'shuf': 'di'}
```

- val_acc: 0.879913


### TN16+PN10+pretrain (di. shuffle, no fine-tuning)

- Dir: 46f970b1c6379f85b7ccc1fe68a8af14

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
preenc=output_enc/9780456c95e7c048e2501106fd40c716
pretrained=False
pseq=True
pseq_embdim=16
pseq_encdim=400
```

- MSEs: 0.17270962312103763, 0.2207709531401928, 0.18877799411561982, 0.21240548011506646, 0.22274311408665057
- MSE: 0.20348143291571344, STDEV: 0.019558496261868393


#### Without PN10

- Dir: 9ae037a97e5283d4651b77444b97dd42
- MSE: 0.3629515767422088, 0.03559513166108524


### TN16+PN10+pretrain (di. shuffle, with fine-tuning)

- Dir: 0aa2570ced889c4e88ae1554253cb412

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
preenc=output_enc/9780456c95e7c048e2501106fd40c716
pretrained=False
pseq=True
pseq_embdim=16
pseq_encdim=400
```

- MSEs: 0.18851345297596922, 0.16698824099978538, 0.1784539630179947, 0.22683654581530616, 0.2222658611700897
- MSE: 0.19661161279582903, STDEV: 0.023851673443049014


#### Without PN10

- Dir: b65364664e64110a43ca71623a066182
- MSE: 0.3654499088033158, 0.032623712329671886


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

- val_acc: 0.732533


### TN16+PN10+pretrain (sent. shuffle, no fine-tuning)

- Dir: 752aa6160e706a6ffe0f91a1e423b40a
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

- MSEs: 0.1708993219855502, 0.188276010211909, 0.16168191047724936, 0.21673573230273174, 0.21641181262379064
- MSE: (0.1908009575202462, STDEV: 0.022710780694247448)	


#### Without PN10

- Dir: f8a2a3674079b877485038b9f59818ff
- MSE: 0.34423311125480527, 0.017422520525839476


### TN16+PN10+pretrain (sent. shuffle, with fine-tuning)

- Dir: 0cb7a2429b6cc73297413c20570c824f

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

- MSEs: 0.17768359446192292, 0.16614793271124353, 0.1609080225191747, 0.20891687420427643, 0.21897310227394462
- MSE: 0.18652590523411244, STDEV: 0.023254677920348094


#### without PN10

- Dir: a2fe2d615652d3170309fc000cd559be
- MSE: 0.34666465929342216, 0.016055677326282502


### PN10

fold_0: MSE: 0.1706966015579623, MAE: 0.3322794187068939
fold_1: MSE: 0.18432975306633015, MAE: 0.33795325969582174
fold_2: MSE: 0.16311654072939488, MAE: 0.33474165883230333
fold_3: MSE: 0.2154137200655532, MAE: 0.36318332401674186
fold_4: MSE: 0.2025787649746826, MAE: 0.36014226675033567

- MSE: 0.1872271

## Argument strength

# TN16

### fold_1:

- clipnorm 0, dropout 0.5 = val_loss: 0.0286
- clipnorm 0, dropout 0.7 = val_loss: 0.0291
- clipnorm 0, dropout 0.9 = val_loss: 0.0292

- clipnorm 5, dropout 0.7 = val_loss: 0.0290
- clipnorm 5, dropout 0.5 = val_loss: 0.0292
- clipnorm 5, dropout 0.9 = val_loss: 0.0290

- clipnorm 3, dropout 0.5 = val_loss: 0.0289
- clipnorm 3, dropout 0.7 = val_loss: 0.0292
- clipnorm 3, dropout 0.9 = val_loss: 0.0291

### fold_3:

- clipnorm 0, dropout 0.5 = val_loss: 0.0240
- clipnorm 3, dropout 0.5 = val_loss: 0.0245
- clipnorm 5, dropout 0.7 = val_loss: 0.0243
- clipnorm 5, dropout 0.9 = val_loss: 0.0243

dir: 2ed64585fdaf6e39bf91918d1a54ca5c

fold_0: MSE: 0.2659966972675082, MAE: 0.41611958146095274
fold_1: MSE: 0.2378322980040582, MAE: 0.40255620002746584
fold_2: MSE: 0.25223326058600803, MAE: 0.42349262118339537
fold_3: MSE: 0.2351862960945587, MAE: 0.4028599011898041
fold_4: MSE: 0.2622018043749034, MAE: 0.4131241583824158

MSE: 0.25069007

# see results:

## TN16_PN10: Organization

dir: output/f3266eb6be217d61a69d21977169665f