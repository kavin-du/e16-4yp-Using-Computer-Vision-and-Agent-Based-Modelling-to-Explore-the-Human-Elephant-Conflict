
## command line arguments
`-e` or `--epochs` ,      type=int, default=50      => number of epochs\
`-b` or `--batches` ,     type=int, default=8       => batch size\
`-r` or `--resolution` ,  type=int, default=1024    => resolution of image SxS\
`-t` or `--testing` ,               default=False   => switch between training and testing\
`-m` or `--model` ,       type=str, default=None    => relative path for the saved model\
`-d` or `--dataset` ,     type=str, default='.'     => relative path for the dataset\

## directory structure for satellite images

parent directory  
    |-> Train                  
    |-> Val\
    |-> Test

_path of the parent directory should be given to the argument '-d'_\
_'Test' folder is not essential when model trains_

## example for model train

_epochs=50, batch size=16, resolution=1024, dataset path='../datasets'_

`python3 train.py -e 50 -b 16 -d ../datasets -r 1024`


## example for model testing with test images

The test images should be in a folder named "Test" as shown in the above directory structure. 

Path for the saved .pt model and the path for the Test dataset should be given as arguments. 

`python3 train.py --testing -m model_path -d dataset_path`

`python3 train.py --testing -m ./saved_models/unet_epoch_13_0.98493.pt -d ../datasets`

The testing results will be saved to a folder named "visualizations". 

You can compare the original image vs the masked image in the same plot. 

![2](https://user-images.githubusercontent.com/59405594/183392572-31674a18-7403-4e80-8f8c-e65e2bc07cfd.jpg)

