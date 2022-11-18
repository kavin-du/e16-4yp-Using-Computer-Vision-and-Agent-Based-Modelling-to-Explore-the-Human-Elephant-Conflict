#!/bin/bash

conda install -y -c zimmf cudatoolkit   # cuda 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html

pip install -r requirements.txt

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  