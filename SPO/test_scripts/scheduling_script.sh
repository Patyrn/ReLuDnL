#!/bin/sh
for load in 1 2 3
do
    for layer_params in  501 901 50501
    do
      sbatch ./slurms/icon_scheduling_l${load}_param0${layer_params}.slurm
    done
done