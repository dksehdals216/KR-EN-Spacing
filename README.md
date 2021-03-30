
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
data processing:

```./prepare_data.sh```

model training & generation: (under ~/mod_nmt_pytorc/nmt_pytorch/)

```train_enkr.sh```

model testing: (under ~/mod_nmt_pytorc/nmt_pytorch/)

```test_enkr.sh```

detokenize: (under ~/mod_nmt_pytorc/nmt_pytorch/)

```nmt_post.sh```

#### Spaced  


### (EN - KR)

#### Original 

#### Spaced 




## Models

## Author
In no particular order:

* Dongmin An (21600397@handong.edu)
* Seulgi Choi (21700729@handong.edu)
* Sojung Hwang (sojeoung536985@gmail.com)
* Youngpyo Kim (menstoo9504@gmail.com)

## Reference

