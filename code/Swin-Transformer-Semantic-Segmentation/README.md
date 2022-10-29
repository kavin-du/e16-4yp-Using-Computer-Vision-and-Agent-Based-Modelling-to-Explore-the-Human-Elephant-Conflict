conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install -c zimmf cudatoolkit   # cuda 11.1
conda install pytorch=1.8.1 torchvision -c pytorch

pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8/index.html

pip install -r requirements.txt