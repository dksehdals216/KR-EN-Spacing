
# KR-EN-Spacing

## Introduction
Bidirectional LSTM-CRF NMT system with Korean spacing in KR-EN, EN-KR
translation

## Project Tree
  * backup
  * data
  * nmt_pytorch
  * mod_nmt_pytorch
  * spacer

## Usage
### (EN - KR)
#### Original 
__data processing:__

```./prepare_data.sh```

__model training & generation: (under ~/mod_nmt_pytorc/nmt_pytorch/)__

```train_enkr.sh```

__model testing: (under ~/mod_nmt_pytorc/nmt_pytorch/)__

```test_enkr.sh```

__detokenize: (under ~/mod_nmt_pytorc/nmt_pytorch/)__

```nmt_post.sh```

#### Spaced  

__data processing:__

```~/spacer/prep_enkr_space.sh```
```~/spacer/main.py```
```~/text_data_gen/post_enkr_space.sh```


### (KR - EN)

#### Original 
__data processing:__

```~/nmt_pytorch/prep_kren_orig.sh```

__model training: (under ~/mod_nmt_pytorc/nmt_pytorch/)__

Edit DATA_DIR Variable inside ~/nmt_pytorch/train_kren.sh to: 
Data/kr-en/processed/original_data

```~/nmt_pytorch/train_kren.sh ```

__model testing: (under ~/mod_nmt_pytorc/nmt_pytorch/)__****

Edit DATA_DIR Variable inside ~/nmt_pytorch/trans_kren.sh to:
Data/kr-en/processed/original_data

```~/nmt_pytorch/trans_kren.sh```



#### Spaced 




## Models

## Author
In no particular order:

* Dongmin An (21600397@handong.edu)
* Seulgi Choi (21700729@handong.edu)
* Sojung Hwang (sojeoung536985@gmail.com)
* Youngpyo Kim (menstoo9504@gmail.com)

## Reference

