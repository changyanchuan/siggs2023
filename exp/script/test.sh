#!/bin/bash

exe_file='../../main.py'
methods=(unet)
seeds=(2000 2001 2002 2003 2004)
cel_weights=(500 400 300 200 100)

function evalcmd () {
    echo $1
    eval $1
    sleep 1.5s
}


for cel_weight in ${cel_weights[*]}
do
    ts=$(date +%s%7N)

    wholecommand="python ${exe_file} --training_cel_weight ${cel_weight} --dumpfile_uniqueid ${ts}"
    evalcmd "$wholecommand"
    done
done
