# #!/bin/bash
# # Created by the University of Melbourne job script generator for SLURM
# # Thu Aug 06 2020 16:14:44 GMT+1000 (Australian Eastern Standard Time)
#
# # Partition for the job:
# #SBATCH --partition=physical
#
# # Multithreaded (SMP) job: must run on one node
# #SBATCH --nodes=1
#
# # The name of the job:
# #SBATCH --job-name="knap"
#
# # The project ID which this job should run under:
# #SBATCH --account="punim1171"
#
# # Maximum number of tasks/CPU cores used by the job:
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
#
#
# # Send yourself an email when the job:
# # aborts abnormally (fails)
# #SBATCH --mail-type=FAIL
# # begins
# #SBATCH --mail-type=BEGIN
# # ends successfully
# #SBATCH --mail-type=END
#
# # Use this email address:
# #SBATCH --mail-user=aguler@student.unimelb.edu.au
#
# # The maximum running time of the job in days-hours:mins:sec
# #SBATCH --time=4-0:0:00
#
# # check that the script is launched with sbatch
# if [ "x$SLURM_JOB_ID" == "x" ]; then
#    echo "You need to submit your job to the queuing system with sbatch"
#    exit 1
# fi
#
# # Run the job from the directory where it was launched (default)
#
# # The modules to load:
# module load gcc/8.3.0
# module load openmpi/3.1.4
# module load python/3.7.4
# module load scikit-learn/0.23.1-python-3.7.4
# module load gurobi/9.0.0
# module load numpy/1.17.3-python-3.7.4
# module load matplotlib/3.2.1-python-3.7.4
# module load pytorch/1.5.1-python-3.7.4
#
# export GRB_LICENSE_FILE=/usr/local/easybuild/software/Gurobi/gurobi.lic
# time python weighted_knapsack_batch_code1.py iconel12k1.txt
file_folder = 'slurms/'
def script1():
    a = '#!/bin/bash \n'+\
        '# Created by the University of Melbourne job script generator for SLURM\n' +\
        '# Thu Aug 06 2020 16:14:44 GMT+1000 (Australian Eastern Standard Time)'
    print(a)

def scheduling_script(load, divider, layer_params):
    a = """#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Thu Aug 06 2020 16:14:44 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=physical

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="schedul{}"

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
#SBATCH --time=4-0:0:00

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

export GRB_LICENSE_FILE=/usr/local/easybuild/software/Gurobi/gurobi.lic
time python /home/aguler/relu_dnl/Relu_DNL/test_scripts/icon_scheduling.py {} {} {}""".format(load, load, divider, " ".join(map(str,layer_params)))

    file_name = file_folder + "icon_scheduling_l{}_div{}_param0{}.slurm".format(load, divider, "0".join(map(str, layer_params)))
    with open(file_name, 'w') as f:
        f.write(a)

def scheduling_shell(loads, dividers, layer_params_list):
    loads_str=" ".join(map(str, loads))
    dividers_str =" ".join(map(str, dividers))
    layer_params_str = ""
    for layer_params in layer_params_list:
        layer_params_str += " " + "0".join(map(str, layer_params))
    a="""#!/bin/sh
for load in {}
do
  for divider in {}
  do
    for layer_params in {}
    do
      sbatch ./slurms/icon_scheduling_l${{load}}_div${{divider}}_param0${{layer_params}}.slurm
    done
  done
done""".format(loads_str, dividers_str, layer_params_str)
    with open("scheduling_script.sh", 'w') as f:
        f.write(a)

def knapsack_shell(capacities, dividers, layer_params_list):
    capacities_str=" ".join(map(str, capacities))
    dividers_str =" ".join(map(str, dividers))
    layer_params_str = ""
    for layer_params in layer_params_list:
        layer_params_str += " " + "0".join(map(str, layer_params))
    a="""#!/bin/sh
for load in {}
do
  for divider in {}
  do
    for layer_params in {}
    do
      sbatch ./slurms/icon_knapsack_c${{load}}_div${{divider}}_param0${{layer_params}}.slurm
    done
  done
done""".format(capacities_str, dividers_str, layer_params_str)
    with open("knapsack_script.sh", 'w') as f:
        f.write(a)
def knapsack_script(capacity, divider, layer_params):
    a = """#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Thu Aug 06 2020 16:14:44 GMT+1000 (Australian Eastern Standard Time)

# Partition for the job:
#SBATCH --partition=physical

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="knap{}"

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
#SBATCH --time=4-0:0:00

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

export GRB_LICENSE_FILE=/usr/local/easybuild/software/Gurobi/gurobi.lic
time python /home/aguler/relu_dnl/Relu_DNL/test_scripts/icon_knapsack.py {} {} {}""".format(capacity,capacity, divider, " ".join(map(str, layer_params)))

    file_name = file_folder + "icon_knapsack_c{}_div{}_param0{}.slurm".format(capacity, divider, "0".join(map(str, layer_params)))
    with open(file_name, 'w') as f:
        f.write(a)


if __name__ == '__main__':
    #Scheduling

    loads = [30,31,32,33,34,35,36,37,38,39,400,40,41,42,43,44,45,46,47,48,500,501,50,51,52,53,54,55,56,57]
    dividers = [1,5,10]
    layer_params_list =[[1],[5,1],[9,1],[5,5,1]]
    for load in loads:
        for divider in dividers:
            for layer_params in layer_params_list:
                scheduling_script(load, divider, layer_params)
    scheduling_shell(loads,dividers,layer_params_list)

    #Knapsack

    capacities = [12, 24, 48, 72, 96, 120, 144, 172, 196, 220]
    dividers = [1,5,10]
    layer_params_list =[[1],[5,1],[9,1],[5,5,1]]
    for c in capacities:
        for divider in dividers:
            for layer_params in layer_params_list:
                knapsack_script(c, divider, layer_params)

    knapsack_shell(capacities,dividers,layer_params_list)