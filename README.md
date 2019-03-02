# multi-modal-action-classification

Action classification based on deep learning with multi modal datasets.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Skeleton](#skeleton)
* [Training](#training)

## Prerequisites

Dependencies is required as follow:

- PyTorch 1.0
- tensorboardX 1.6 and tensorboard 1.12

## Skeleton

Implementation of HCN (**Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation**. Hikvision Research Institute. IJCAI 2018. [[arXiv](http://arxiv.org/pdf/1804.06055.pdf)])

#### preprocess

```bash
# change the access permissions
chmod 777 preprocess_script.sh
# preprocess dataset using arguments
# 1. PKU-MMD version, 2. sequence length
./preprocess_script.sh 1 256
```

## Training

**skeleton**

```bash
# if has GPUs, add arguments --gpu
python main.py --sequence-length <SEQUENCE_LENGTH>
```

