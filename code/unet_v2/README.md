
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

## example

_epochs=50, batch size=16, resolution=1024, dataset path='../datasets'_

`python3 train.py -e 50 -b 16 -d ../datasets -r 1024`