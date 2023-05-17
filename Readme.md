# Multi-scale Information Distillation Network for Efficient Image Super-Resolution
This repository is for MSID introduced in the paper.
## Dependenices
* python 3.7
* pytorch 1.10
* NVIDIA GPU + CUDA

## Models
The pretrained models can be found in 'models/'. 
## Data preparing for training
To fully train the models, please download [DIV2K training dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K training dataset](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), and place them in "data/Datasets/Train/DF2K". 

## Data preparing for testing
To fully evaluate the models, please download the following datasets:
* Download [Set5 datasets](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip) into the path "data/Datasets/Test/Set5". 
* Download [Set14 datasets](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip) into the path "data/Datasets/Test/Set14". 
* Download [BSD100 datasets](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip) into the path "data/Datasets/Test/BSD100".
* Download [Urban100 datasets](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip) into the path "data/Datasets/Test/Urban100". 
* Download [Manga109 datasets](http://www.manga109.org/ja/index.html) into the path "data/Datasets/Test/Manga109". 

## Settings in option.py
* For Training
```
    '-train' == 'train'
```
* For Testing
```
    '-train' == 'test'
```

* Set the "scale" for x2, x3, or x4 upscaling:
````
  '-scale' == 4
````