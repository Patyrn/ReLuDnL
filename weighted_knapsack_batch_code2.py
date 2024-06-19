from Experiments import test_knapsack_weighted

"""
Example Dnl experiments on weighted and unit knapsack problems. 
Test boolean (boolean array): determines the variations of dnl used. The order is [Exhaustive, Exhaustive_max, Dnl, dnl_max, dnl_greedy]. exhaustive, dnl and dnl_greedy are used in the paper.
for dnl_greedy choose test boolean = [0,0,0,0,1]

Dependencies:
gcc/8.3.0
openmpi/3.1.4
python/3.7.4
scikit-learn/0.23.1-python-3.7.4
gurobi/9.0.0
numpy/1.17.3-python-3.7.4
matplotlib/3.2.1-python-3.7.4

"""

file_folder = "Tests/Icon_knapsack/ReLuDNL"
prefix = 'Iconknap_c'
layer_params_list = [[1],[5,1],[9,1],[5,5,1]]
param_dividers = [10,5,1]
capacities = [120, 144, 172, 196, 220]
kfolds = [0, 1, 2,3,4]
test_boolean = [0,1]

for param_divider in param_dividers:
    for layer_params in layer_params_list:
        test_knapsack_weighted(max_step_size_magnitude=0, min_step_size_magnitude=-1, capacities=capacities, dnl_epoch=1,
                               regression_epoch=20, core_number=8, learning_rate=0.01, dnl_learning_rate=1, mini_batch_size=32,
                               n_iter=2,is_save=True, file_name_prefix_phrase=prefix, kfolds=kfolds, dnl_batch_size=None, test_boolean=test_boolean)


