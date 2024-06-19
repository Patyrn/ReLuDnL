#!/bin/sh
for load in 30 31 32 33 34 35 36 37 38 39 400 40 41 42 43 44 45 46 47 48 500 501 50 51 52 53 54 55 56 57
do
  for divider in 1 5 10
  do
    for layer_params in  1 501 901 50501
    do
      sbatch ./slurms/icon_scheduling_l${load}_div${divider}_param0${layer_params}.slurm
    done
  done
done