# RAM-CD: Dynamic Deformation Alignment with Rank-Guided Enhancement for Remote Sensing Change Detection

Here, we provide the pytorch implementation of the paper: "RAM: Rank-Aware Multi-Scale Feature Alignment for Remote Sensing Change Detection".

<img src="./images/RAM-CD.jpg" alt="RAM-CD" style="zoom: 33%;" />

## Requirements

```
Python 3.10
pytorch 2.4.1
torchvision 0.19.1
```

## Installation

Clone this repo:

```shell
git clone https://github.com/joybo-08/RAM-CD.git
cd RAM_main
```

## Train

You can run a train to get started as follows:

```
python main_cd.py
```

The detailed parameter settings is as follows:

```cmd
gpus=0
checkpoint_root=checkpoints 
data_name=LEVIR  # dataset name 

img_size=256
batch_size=8
lr=0.01
max_epochs=200  #training epochs
#base_resnet18
#base_transformer_pos_s4_dd8
#base_transformer_pos_s4_dd8_dedim8
lr_policy=linear

split=train  # training txt
split_val=val  #validation txt
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}
```

## Evaluate

Then, run a evaluate to get started as follows:

```
python eval_cd.py
```

The detailed parameter settings is as follows:

```cmd
gpus=0
data_name=LEVIR # dataset name
split=test # test.txt
project_name=BIT_LEVIR # the name of the subfolder in the checkpoints folder 
checkpoint_name=best_ckpt.pt # the name of evaluated model file 

python eval_cd.py --split ${split} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}
```

## Dataset Preparation

### Data structure

```
"""
Change detection data set with pixel-level binary labels；
list
train
  ├─A
  ├─B
  ├─label
  └─list
val
  ├─A
  ├─B
  ├─label
  └─list
test
  ├─A
  ├─B
  ├─label
  └─list

"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.

### Data Download 

LEVIR-CD: https://justchenhao.github.io/LEVIR/

WHU-CD: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html



