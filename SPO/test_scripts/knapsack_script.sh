#!/bin/sh
for layer_params in  501 901 50501
do
    sbatch ./slurms/icon_knapsack_param0${layer_params}.slurm
done