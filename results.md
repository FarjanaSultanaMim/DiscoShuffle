BEST:

Earlystopping = 15, Bidirectional_GRU

# Organization

## Baselines
### TN16

Best hyperparameter based on fold no. 0.

#### Final results

Dir: 5c078fb2691f6e000ef7ccdf0c02a7c8

```
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea --dropout 0.5 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 10 --meanovertime \
    --pre-trained --fix-embedding
```

- MSEs: 0.3749731966880151, 0.3154149795353907, 0.3813750535541138, 0.30842621931571246, 0.35948690976970027
- MSE: 0.34793527177258643, STDEV: 0.030335600745584456


### TN16+PN10

Best hyperparameter based on fold no. 0.

#### Final results

Dir: e885c38c446af586e887fb790dd90930

```
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea --dropout 0.75 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 10 --meanovertime \
    --pre-trained --fix-embedding \
    --persing-seq --pseq-embedding-dim 16 --pseq-encoder-dim 400
```

- MSEs: 0.1839222235003539, 0.18074305694272838, 0.2015612465336829, 0.21749697897579662, 0.21812192476768963
- MSE: 0.20036908614405027, STDEV: 0.015909931184509462


## Document encoder pretraining

### With DI shuffling

#### Pretraining

```
{'aggr_grudim': '300', 'att': 'False', 'clipnorm': '5.0', 'dropout': '0.7', 'emb_dim': '50', 'emb_fix': 'False', 'enc_fix': 'False', 'encdim': 'None', 'model_type': 'nea', 'mot': 'True', 'pretrained': 'False', 'shuf': 'di'}
```

- val_acc: 0.879913


#### TN16+PN10+pretrain (di. shuffle, no fine-tuning)

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


##### Without PN10

- Dir: 9ae037a97e5283d4651b77444b97dd42
- MSE: 0.3629515767422088, 0.03559513166108524


#### TN16+PN10+pretrain (di. shuffle, with fine-tuning)

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


##### Without PN10

- Dir: b65364664e64110a43ca71623a066182
- MSE: 0.3654499088033158, 0.032623712329671886


### With sentence shuffling

#### Pretraining

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


#### TN16+PN10+pretrain (sent. shuffle, no fine-tuning)

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


##### Without PN10

- Dir: f8a2a3674079b877485038b9f59818ff
- MSE: 0.34423311125480527, 0.017422520525839476


#### TN16+PN10+pretrain (sent. shuffle, with fine-tuning)

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


##### without PN10

- Dir: a2fe2d615652d3170309fc000cd559be
- MSE: 0.34666465929342216, 0.016055677326282502


# Argument Strength

## TN16

```
python src/train.py \
    --fold {} \
    --score-type {} \
    --model-type nea --dropout 0.25 \
    --embedding-dim 50 --aggregation-grudim 300 \
    --gradientclipnorm 5 --meanovertime \
    --pre-trained --fix-embedding
```

- Dir: c223f9171d2065520643cf6d654dc138
- MSEs: 0.26188567369116356, 0.23913804768089647, 0.2773715832152496, 0.23792502091095286, 0.2588275069897034
- MSE: 0.2550295664975932, STDEV: 0.014870790959933896


## Document encoder pretraining

### With DI shuffling

#### Pretraining

```
{'aggr_grudim': '300', 'att': 'False', 'clipnorm': '5.0', 'dropout': '0.7', 'emb_dim': '50', 'emb_fix': 'False', 'enc_fix': 'False', 'encdim': 'None', 'model_type': 'nea', 'mot': 'True', 'pretrained': 'False', 'shuf': 'di'}
```

- val_acc: 0.879913


#### TN16+pretrain (di. shuffle, no fine-tuning)

- Dir: cee23dc2ed54aed0911230d84151441e

- MSEs: 0.25664800333384447, 0.23941677197983893, 0.2723700790054423, 0.23736549721363503, 0.26929889274507046
- MSE: 0.2550198488555663, STDEV:0.014578783265718775


#### TN16+pretrain (di. shuffle, with fine-tuning)

- Dir: b1809650ecdefed27f07e32a05dc3ade

- MSEs: 0.2592230096534951, 0.23955059426323458, 0.259447455157723, 0.23680760373543905, 0.267192892992712
- MSE: 0.25244431116052074, STDEV: 0.012027219127379462


### With sentence shuffling

#### Pretraining

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


#### TN16+pretrain (sent. shuffle, no fine-tuning)

- Dir: fcbeada6ec3cb1984984fe3ec9cb664e

- MSEs: 0.2553112997422369, 0.2392242413397571, 0.2528419103068555, 0.2315592930770336, 0.26730756532314587
- MSE: 0.24924886195780577, STDEV: 0.01256338952106033


#### TN16+pretrain (sent. shuffle, with fine-tuning)

- Dir: 51a2e8727c1fe1fa27d190f879ce078d

- MSEs: 0.26496048231386626, 0.23985156913924272, 0.2462020482008063, 0.23528099814833667, 0.26786079507282806
- MSE: 0.250831178575016, STDEV: 0.013216912233143372
