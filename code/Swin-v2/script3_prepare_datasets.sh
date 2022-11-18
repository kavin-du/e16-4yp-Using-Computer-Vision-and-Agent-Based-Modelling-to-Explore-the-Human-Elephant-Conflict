#!/bin/bash

# downloading zip files
FOLDER_NAME=dataset_zips

mkdir $FOLDER_NAME

wget -nc https://zenodo.org/record/5706578/files/Test.zip -P $FOLDER_NAME

wget -nc https://zenodo.org/record/5706578/files/Train.zip -P $FOLDER_NAME

wget -nc https://zenodo.org/record/5706578/files/Val.zip -P $FOLDER_NAME

# converting the datasets
python tools/convert_datasets/loveda.py $FOLDER_NAME

