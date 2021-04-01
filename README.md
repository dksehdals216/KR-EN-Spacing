
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

__train__:
```~/mod_nmt_pytorch/nmt_pytorch/train_enkr.sh```

__model_testing:__:
```~/mod_nmt_pytorch/nmt_pytorch/trans_enkr_space.sh```

__detokenize: (Change src and tgt accordingly)__

```~/mod_nmt_pytorch/nmt_pytorch/nmt_post.sh```



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
__data processing:__
in ```~/spacer/config.py```
set variable test_data as:
```~/data/kr-en/processed/spacing_data/aihub.train.kr```

```~/spacer/main.py```
```~/text_gen_data/post_kren_space.sh```

__model training:__
```~/nmt_pytorch/train_spaced_kren.sh```

__model testing:__
```~/nmt_pytorch/trans_space_kren.sh```

## Models

## Author
In no particular order:

* Dongmin An (21600397@handong.edu)
* Seulgi Choi (21700729@handong.edu)
* Sojung Hwang (sojeoung536985@gmail.com)
* Youngpyo Kim (menstoo9504@gmail.com)

## Reference

