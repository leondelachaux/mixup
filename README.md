## Introduction
This repository contains code from https://github.com/facebookresearch/mixup-cifar10.

## Reference
```
@article{
zhang2018mixup,
title={mixup: Beyond Empirical Risk Minimization},
author={Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz},
journal={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=r1Ddp1-Rb},
}
```

## Requirements and Installation
* A computer running macOS or Linux
* For training new models, you'll also need a NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6
* A [PyTorch installation](http://pytorch.org/)

## Training
Use `python train2.py` to train a new model.
I ran the code on a kaggle notebook, but it also works on Google Colab.
First, I import the CIFAR-10-C and CIFAR-100-C datasets with

```
!mkdir -p ./data/cifar
!curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
!curl -O https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
!tar -xvf CIFAR-100-C.tar -C data/cifar/
!tar -xvf CIFAR-10-C.tar -C data/cifar/
```
Then I clone the repository with
```
!git clone https://github.com/leondelachaux/mixup
```
```
import sys
sys.path.append('mixup')
```
and finally I can just choose the model I want to train as well as the dataset with
```
!python mixup/train2.py --dataset cifar10 --model allcanv --lr 0.1 --epoch 200 --decay 1e-4
```
