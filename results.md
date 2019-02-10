# Organization

## NEA (without punctuation)

dir: output/4628ce2b6dfa7172836b7681526120a5

`aggr_grudim=300
att=False
clipnorm=5.0
di_aware=False
dropout=0.7
elmo=False
emb_dim=50
emb_fix=True
enc_fix=False
encdim=None
model_type=nea
mot=True
only_pseq=False
preenc=None
pretrained=True
prompt=False
pseq=False
pseq_conv_encdim=None
pseq_embdim=None
pseq_encdim=None
punct=False
score_type=Organization
seed=None
ulstm=False`

- MSE: 0.1878023529089173, MAE: 0.32671822130680084
- MSE: 0.2082130334434329, MAE: 0.34339458195131217
- MSE: 0.45734209973963047, MAE: 0.450471950407645
- MSE: 0.2280901504218698, MAE: 0.369896272521707
- MSE: 0.37121416921440753, MAE: 0.4541762804985046

###### Avg: 0.2905323611456516 , STDEV: 0.11786008551326131 



## NEA (with punctuation)

dir: output/fc20c3e804de8d8cd34fa4f6a91f69a8

`aggr_grudim=300
att=False
clipnorm=5.0
di_aware=False
dropout=0.7
elmo=False
emb_dim=50
emb_fix=True
enc_fix=False
encdim=None
model_type=nea
mot=True
only_pseq=False
preenc=None
pretrained=True
prompt=False
pseq=False
pseq_conv_encdim=None
pseq_embdim=None
pseq_encdim=None
punct=True
score_type=Organization
seed=None
ulstm=False`

- MSE: 0.18260250342412412, MAE: 0.3351481992006302
- MSE: 0.22000765670769332, MAE: 0.3563660377293677
- MSE: 0.4281403508514251, MAE: 0.43943847589824925
- MSE: 0.21872622732353686, MAE: 0.35528283866483773
- MSE: 0.20076873216368318, MAE: 0.35255627453327176

###### Avg: 0.25004909409409254 , STDEV: 0.10071953924309758


## NEA (with punctuation & prompt)

dir: output/cd311c50d8f814bb6ab99613922e73b7

`aggr_grudim=300
att=False
clipnorm=5.0
di_aware=False
dropout=0.7
elmo=False
emb_dim=50
emb_fix=True
enc_fix=False
encdim=None
model_type=nea
mot=True
only_pseq=False
preenc=None
pretrained=True
prompt=True
pseq=False
pseq_conv_encdim=None
pseq_embdim=None
pseq_encdim=None
punct=True
score_type=Organization
seed=None
ulstm=False`

- MSE: 0.1834161630576773, MAE: 0.3334080243110657
- MSE: 0.2344542887788437, MAE: 0.36870682595381093
- MSE: 0.1874720934144812, MAE: 0.3433224016161107
- MSE: 0.2166135046898376, MAE: 0.35784506738482424
- MSE: 0.20066742294185325, MAE: 0.35274820268154145

###### Avg: 0.2045246945765386 STDEV: 0.021172858763100256 



## NEA+pretrained (sentence shuffle, fixed encoder, encoder:no_pre_embed)

dir: output/60bf422378be9e1b3987cc17e066ed9a

`aggr_grudim=300
att=False
clipnorm=5.0
di_aware=False
dropout=0.7
elmo=False
emb_dim=50
emb_fix=True
enc_fix=True
encdim=None
model_type=nea_aft_pretrain
mot=True
only_pseq=False
preenc=output_enc/c67b32bab136bdc45d6a32950b6a2d9a
pretrained=False
prompt=True
pseq=False
pseq_conv_encdim=None
pseq_embdim=None
pseq_encdim=None
punct=True
score_type=Organization
seed=None
ulstm=False`


- MSE: 0.37731602692763744, MAE: 0.4343016469478607
- MSE: 0.3187007523967158, MAE: 0.39297205535926627
- MSE: 0.400095211678795, MAE: 0.42618024645753166
- MSE: 0.3199420830772437, MAE: 0.4386341049896544
- MSE: 0.37504054243561996, MAE: 0.45764305114746096

######


## NEA+pretrained (sentence shuffle, tuned, encoder:no_pre_embed)

dir: 


-
-
-
-
-

######




## NEA+PN10 (without punctuation)

dir: output/2eb663e7efb7671f4f1864d696f14f11

`aggr_grudim=300
att=False
clipnorm=5.0
di_aware=False
dropout=0.7
elmo=False
emb_dim=50
emb_fix=True
enc_fix=False
encdim=None
model_type=nea
mot=True
only_pseq=False
preenc=None
pretrained=True
prompt=False
pseq=True
pseq_conv_encdim=None
pseq_embdim=16
pseq_encdim=200
punct=False
score_type=Organization
seed=None
ulstm=False`

- MSE: 0.16936961639231604, MAE: 0.32286689460277557
- MSE: 0.223380031285641, MAE: 0.36362135944081775
- MSE: 0.17517329934983522, MAE: 0.34519909033134805
- MSE: 0.18084237743151887, MAE: 0.333873290327651
- MSE: 0.1804171990589385, MAE: 0.33374626100063326

###### Avg: 0.18583650470364993 , STDEV: 0.021498191366771304 

## NEA+PN10 (with punctuation)

dir: output/7820467193fbfa10e5cfbda5c5658528

`aggr_grudim=300
att=False
clipnorm=5.0
di_aware=False
dropout=0.7
elmo=Falseemb_dim=50
emb_fix=True
enc_fix=False
encdim=None
model_type=nea
mot=True
only_pseq=False
preenc=None
pretrained=True
prompt=False
pseq=True
pseq_conv_encdim=None
pseq_embdim=16
pseq_encdim=200
punct=True
score_type=Organization
seed=None
ulstm=False`

- MSE: 0.16871184032246306, MAE: 0.3311817812919617
- MSE: 0.18711309118910605, MAE: 0.3464867072318917
- MSE: 0.1887715425230061, MAE: 0.3578518729897874
- MSE: 0.17515299104798732, MAE: 0.32116006322168
- MSE: 0.17511622015883574, MAE: 0.3290806007385254

###### Avg: 0.17897313704827966 , STDEV: 0.008617295647602958



## NEA+PN10 (with punctuation & prompt)

dir: output/c790a61c043dab53a0346392a4159c23

`aggr_grudim=300
att=False
clipnorm=5.0
di_aware=False
dropout=0.7
elmo=False
emb_dim=50
emb_fix=True
enc_fix=False
encdim=None
model_type=nea
mot=True
only_pseq=False
preenc=None
pretrained=True
prompt=True
pseq=True
pseq_conv_encdim=None
pseq_embdim=16
pseq_encdim=200
punct=True
score_type=Organization
seed=None
ulstm=False`

- MSE: 0.1678823151775316, MAE: 0.32247899651527406
- MSE: 0.19288021450797263, MAE: 0.3457663587076747
- MSE: 0.15539073949973203, MAE: 0.3195635875066121
- MSE: 0.17211925494047242, MAE: 0.3175252272715023
- MSE: 0.19781123872090015, MAE: 0.3490826612710953

###### Avg: 0.17721675256932176  STDEV: 0.01774096917985595 




# Argument Strength

## NEA (with punctuation & prompt)

dir: output/e24ec63ee9a71e3a189976fd02eeb9bd

`aggr_grudim=300
att=False
clipnorm=5.0
di_aware=False
dropout=0.7
elmo=False
emb_dim=50
emb_fix=True
enc_fix=False
encdim=None
model_type=nea
mot=True
only_pseq=False
preenc=None
pretrained=True
prompt=True
pseq=False
pseq_conv_encdim=None
pseq_embdim=None
pseq_encdim=None
punct=True
score_type=ArgumentStrength
seed=None
ulstm=False`

- MSE: 0.24380982841340426, MAE: 0.4071388244628906
- MSE: 0.22947361282695625, MAE: 0.39474806785583494
- MSE: 0.24507777364546882, MAE: 0.3966338586807251
- MSE: 0.2416915805948969, MAE: 0.40962729692459104
- MSE: 0.2643505339675244, MAE: 0.4103888630867004

###### Avg: 0.24488066588965013 STDEV: 0.012528057076335102



# Pretrained encoder 

## sentence-shuffling

dir: output_enc/c67b32bab136bdc45d6a32950b6a2d9a

`aggr_grudim=300
att=False
clipnorm=0.0
dropout=0.7
emb_dim=50
emb_fix=False
enc_fix=False
encdim=None
model_type=nea
mot=True
pretrained=False
punct=True
shuf=sentence
ulstm=False`

- train_acc: 0.8651
- val_acc: 0.8232


## Sentence Shuffling (pretrained-embedding)

dir: output_enc/dcca870a2437e232b1c732bd5d4b405b

`aggr_grudim=300
att=False
clipnorm=0.0
dropout=0.7
emb_dim=50
emb_fix=False
enc_fix=False
encdim=None
model_type=nea
mot=True
pretrained=True
punct=True
shuf=sentence
ulstm=False`

- train_acc: 0.8291
- val_acc: 0.8143



# Hyparparameter tuning

## Organization

### NEA

- fold1: 

    best hp:
    dir:
    
- fold2:
     
     best hp:
    dir:
    
- fold3:

     best hp:
    dir:
    
- fold4:
    
     best hp:
    dir:
    
- fold5:
    
     best hp:
    dir:
    












- fold1: 

    best hp:
    dir:
    
- fold2:
     
     best hp:
    dir:
    
- fold3:

     best hp:
    dir:
    
- fold4:
    
     best hp:
    dir:
    
- fold5:
    
     best hp:
    dir: