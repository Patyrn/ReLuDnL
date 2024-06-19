#!/bin/sh
for load in 12 24 48 72 96 120 144 172 196 220
do
  for divider in 1 5 10
  do
    for layer_params in  1 501 901 50501
    do
      sbatch ./slurms/icon_knapsack_c${load}_div${divider}_param0${layer_params}.slurm
    done
  done
done