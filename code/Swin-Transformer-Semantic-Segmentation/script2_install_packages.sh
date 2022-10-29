#!/bin/bash

conda install -y -c zimmf cudatoolkit   # cuda 11.1
conda install -y pytorch=1.8.1 torchvision -c pytorch

pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8/index.html

pip install -r requirements.txt

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .  