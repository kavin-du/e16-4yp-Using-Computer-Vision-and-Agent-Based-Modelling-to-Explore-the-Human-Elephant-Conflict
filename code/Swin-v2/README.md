Check inference in Colab [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/kavin-du/e16-4yp-Using-Computer-Vision-and-Agent-Based-Modelling-to-Explore-the-Human-Elephant-Conflict/blob/swin_loveda/code/Swin-Transformer-Semantic-Segmentation/demo/inference_demo.ipynb)

1. Run the following script to create environments using Conda. Enter the desired name for the env. 
```console
bash script1_create_env.sh
```

2. Restart your shell and activate the above env. 
```console
conda activate <ENV_NAME>
```

3. Run the 2nd script to install the Python packages. 
```console
bash script2_install_packages.sh
```

4. Run the 3rd script to download the dataset, prepare the directories and extract the data. 
```console
bash script3_prepare_datasets.sh
```

5. Run the 4th script to download pretrained models. 
```console
bash script4_download_models.sh
```

We are using the pre-trained model `swin_tiny_patch4_window7_224_22k`.
If you want to try with a different pre-trained model you may download it from here. 
[https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models)


5. To train with a pre-trained model: 
```python
# single-gpu training - not working for now/ will update soon
python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train an UPerNet model with a Swin-T backbone and 8 GPUs, run:

```python
tools/dist_train.sh configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_loveda.py 8 --cfg-options model.pretrained=swin_tiny_patch4_window7_224_22k_converted.pth

```

*To train Swin-T MoBY SSL backbone using 8 GPUs, run:
# needs to be updated

```python
tools/dist_train.sh configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_loveda.py 8 --options model.pretrained=moby_swin_t_300ep_pretrained.pth

```