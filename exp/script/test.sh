#!/bin/bash

exe_file='../../main.py'
methods=(unet) # TODO
seeds=(2000 2001 2002 2003 2004)

function evalcmd () {
    echo $1
    eval $1
    sleep 1.5s
}


for seed in ${seeds[*]}
do
    ts=$(date +%s%7N)

    wholecommand="python ${exe_file} --seed ${seed} --dumpfile_uniqueid ${ts}"
    evalcmd "$wholecommand"
    done
done
