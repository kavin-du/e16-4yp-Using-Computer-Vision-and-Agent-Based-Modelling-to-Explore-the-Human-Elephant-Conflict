#!/bin/bash

printf "%s" "Enter conda environment name: "
read ENV_NAME

SHELL_NAME=`basename "$SHELL"`

printf "Creating the environment $ENV_NAME\n"
conda create -n $ENV_NAME python=3.7 -y
conda init $SHELL_NAME

printf "\n\n\n=====================================\n"
printf "Environment created. Please restart the shell.\n"
printf "Then activate the env using:\n"
printf "conda activate <ENV_NAME>\n"


