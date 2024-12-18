# AirRadar

This repo is the implementation of our manuscript entitled AirRadar: Inferring Nationwide Air Quality in
China with Deep Neural Networks. The code is based on Pytorch 2.1.0, and tested on Ubuntu 16.04 with a NVIDIA GeForce RTX 2080Ti GPU with 12 GB memory. 

In this study, we present a novel deep network named AirRadar to collectively infer nationwide air quality in China.

## Requirements

AirFormer uses the following dependencies: 

* [Pytorch 2.1.0] and its dependencies
* Numpy and Scipy
* CUDA 12.2, cuDNN.


## Folder Structure
We list the code of the major modules as follows:
- The main function to train/test our model: [click here](experiments/airRadar/main.py).
- The source code of our model: [click here](src/models/airRadar.py).
- The trainer/tester: [click here](src/trainers/airRadar_stochastic_trainer.py)/[click here](src/trainers/airRadar_trainer.py)/[click here](src/base/trainer.py)
- Data preparation and preprocessing are located at [click here](src/utils/helper.py).
- Metric computations: [click here](src/utils/metrics.py).

## Arguments
We introduce some major arguments of our main function here.

Training settings:
- mode: indicating the mode, e.g., training or test
- gpu: using which GPU to train our model
- seed: the random seed for experiments
- dataset: which dataset to run
- base_lr: the learning rate at the beginning
- lr_decay_ratio: the ratio of learning rate decay
- batch_size: training or testing batch size
- mask_rate: the mask rate of all nodes
- horizon: the length of future steps
- input_dim: the dimension of inputs
- max_epochs: the maximum of training epochs
- patience: the patience of early stopping
- save_preds: whether to save prediction results

Model hyperparameters:
- n_hidden: hidden dimensions in CT-MSA and DS-MSA
- dropout: dropout rate
- dartboard: which dartboard partition to use. 0: 50-200, 1: 50-200-500, 2: 50, 3: 25-100-250.
- context_num: The number of context.
- block_num : The numbef of Distance-Aware Integrator blocks


## Model Training
Before running our code, please add the path of this repo to PYTHONPATH.
```
export PYTHONPATH=$PYTHONPATH:"the path of this repo"
```

The following examples are conducted on the tiny dataset:
* Example (AirRadar with default setting):
```
python ./experiments/airRadar/main.py --mode train --gpu 0 --dataset AIR_TINY
```

## Model Test
To test above trained models, you can use the following command to run our code:
* Example (AirRadar with default setting):
```
python ./experiments/airRadar/main.py --mode test --gpu 0 --dataset AIR_TINY
```
