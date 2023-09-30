#!/bin/bash

exe_file='../../main.py'
methods=(unet)
seeds=(2000 2001 2002 2003 2004)
cel_weights=(300)
training_u2net_thresholds=(0.8 0.5 0.4 0.3)
# cp_files=(16959247083486106 16959333507227016 16959419795616990 16959506199033390 16959592432453673 16959678773259710 16959765192456520)

function evalcmd () {
    echo $1
    eval $1
    sleep 1.5s
}


aug=""
if [[ $1 == "image_augment" ]] ; then
    aug="--image_augment"
fi


i=0
for training_u2net_threshold in ${training_u2net_thresholds[*]}
do
    ts=$(date +%s%7N)
    # cp_file=${cp_files[$i]}
    # ((i=i+1))
    wholecommand="python ${exe_file} --backbone_str u2net --training_u2net_threshold ${training_u2net_threshold} --dumpfile_uniqueid ${ts} ${aug}"
    # wholecommand="python ${exe_file} --backbone_str u2net --training_u2net_threshold ${training_u2net_threshold} --dumpfile_uniqueid ${cp_file} --load_checkpoint"
    evalcmd "$wholecommand"
done
