#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Thu Aug 06 2020 16:14:44 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=physical

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="knap"

# The project ID which this job should run under:
#SBATCH --account="punim1171"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8


# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# Use this email address:
#SBATCH --mail-user=aguler@student.unimelb.edu.au

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-48:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load gcc/8.3.0
module load openmpi/3.1.4
module load python/3.7.4
module load scikit-learn/0.23.1-python-3.7.4
module load gurobi/9.0.0
module load numpy/1.17.3-python-3.7.4
module load matplotlib/3.2.1-python-3.7.4
module load pytorch/1.5.1-python-3.7.4

export GRB_LICENSE_FILE=/usr/local/easybuild/software/Gurobi/gurobi.lic
time python weighted_knapsack_batch_code2.py iconel12k1.txt

