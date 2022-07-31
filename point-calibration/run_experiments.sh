#!/bin/bash

for experiment in /point-calibration/baseline_model_training/*.sh
do
    echo $experiment
    chmod u+x $experiment
#    sbatch $experiment
    $experiment
    sleep 1
done

for experiment in /point-calibration/recalibration_exp/*.sh
do
    echo $experiment
    chmod u+x $experiment
#    sbatch $experiment
    $experiment
    sleep 1
done

echo "Done"
