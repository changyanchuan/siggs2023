#!/bin/bash

exe_file='../../main.py'
methods=(unet)
seeds=(2000 2001 2002 2003 2004)
cel_weights=(300 100 80 60)
training_celdice_loss_weights=(0.3 0.5 0.7)

function evalcmd () {
    echo $1
    eval $1
    sleep 1.5s
}


for cel_weight in ${cel_weights[*]}
do
    for training_celdice_loss_weight in ${training_celdice_loss_weights[*]}
    do
        ts=$(date +%s%7N)
        wholecommand="python ${exe_file} --backbone_str unet --training_celdice_loss_weight ${training_celdice_loss_weight} --training_cel_weight ${cel_weight} --dumpfile_uniqueid ${ts}"
        evalcmd "$wholecommand"
    done
done