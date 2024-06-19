import copy
import random

import numpy as np
import torch
from joblib._multiprocessing_helpers import mp

from KnapsackSolver import get_opt_params_knapsack
from Relu_DNL.EnergyDataUtil import get_energy_data
from Relu_DNL.Utils import get_train_test_split
from Sampling_Methods import DIVIDE_AND_CONQUER_GREEDY
from Solver import compute_objective_value_single_benchmarks
from relu_ppo import relu_ppo

import matplotlib.pyplot as plt

random.seed(42)
NUMBER_OF_RANDOM_TESTS = 1
random_seeds = [random.randint(0, 100) for p in range(NUMBER_OF_RANDOM_TESTS)]
# random_seeds = [42 for p in range(NUMBER_OF_RANDOM_TESTS)]
global divide_conquer_greedy_time, divide_greedy_profit, divide_profit, divide_conquer_time

def map_parameters_one_problem_set(model, parameter_index, opt_params, benchmark_x, benchmark_y, benchmark_weights):
    temp_model = copy.deepcopy(model)
    number_of_samples = 600
    precision = 10

    first_param = temp_model.get_layer(parameter_index[0][0]).weight.data[parameter_index[0][1]]
    step_size = first_param/precision
    first_param_samples = [first_param - ((number_of_samples/2)*step_size) + i*step_size for i in range(number_of_samples)]

    second_param = temp_model.get_layer(parameter_index[1][0]).weight.data[parameter_index[1][1]]
    step_size = second_param / precision
    second_param_samples = [second_param - ((number_of_samples / 2) * step_size) + i * step_size for i in
                           range(number_of_samples)]

    tovs = np.zeros((number_of_samples,number_of_samples))
    povs = np.zeros((number_of_samples,number_of_samples))
    for first_param_index, first_param_sample in enumerate(first_param_samples):
        for second_param_index, second_param_sample in enumerate(second_param_samples):
            with torch.no_grad():
                temp_model.get_layer(parameter_index[0][0]).weight.data[parameter_index[0][1]] = first_param_sample
                temp_model.get_layer(parameter_index[1][0]).weight.data[parameter_index[1][1]] = second_param_sample

                pred_Y = temp_model.forward(benchmark_x)
                tov,pov = compute_objective_value_single_benchmarks(pred_Y, benchmark_y, benchmark_weights,
                                              opt_params)
                tovs[first_param_index, second_param_index] = tov
                povs[first_param_index, second_param_index] = pov


    # first_param_samples, second_param_samples = np.meshgrid(first_param_samples, second_param_samples)
    sample_points = [first_param_samples, second_param_samples]

    return tovs, povs, sample_points



def map_parameters(file_name_prefix='noprefix', file_folder='', max_step_size_magnitude=0, min_step_size_magnitude=-1,
                   step_size_divider=10, opt_params=None,
                   generate_weight=True, unit_weight=True, is_shuffle=False, print_test=True,
                   test_boolean=None, core_number=7, time_limit=12000, regression_epoch=50, dnl_epoch=3,
                   mini_batch_size=32, dnl_batch_size=None, verbose=False,
                   kfold=0, learning_rate=0.01, hidden_layer_neurons=5, dnl_learning_rate=1, dataset=None,
                   noise_level=0, is_update_bias=False, L2_Lambda=0.001, parameter_index = None, benchmark_no = 0):
    dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=False,
                              kfold=kfold, noise_level=0)

    # combine weights with X first
    # may need to split weights
    for random_test_index, random_seed in zip(range(NUMBER_OF_RANDOM_TESTS), random_seeds):
        train_set, test_set = get_train_test_split(dataset, random_seed=random_seed, is_shuffle=False)

        X_train = train_set.get('X').T
        Y_train = train_set.get('Y').T

        X_val = X_train[0:2880, :]
        Y_val = Y_train[0:2880]

        X_train = X_train[2880:, :]
        Y_train = Y_train[2880:, :]

        benchmarks_train_X = train_set.get('benchmarks_X')
        benchmarks_train_Y = train_set.get('benchmarks_Y')
        benchmarks_weights_train = train_set.get('benchmarks_weights')

        benchmarks_val_X = benchmarks_train_X[0:60]
        benchmarks_val_Y = benchmarks_train_Y[0:60]
        benchmarks_weights_val = benchmarks_weights_train[0:60]

        benchmarks_train_X = benchmarks_train_X[60:]
        benchmarks_train_Y = benchmarks_train_Y[60:]
        benchmarks_weights_train = benchmarks_weights_train[60:]

        # benchmark_number = 1
        train_X = benchmarks_train_X
        train_Y = benchmarks_train_Y
        train_weights = benchmarks_weights_train

        val_X = benchmarks_val_X
        val_Y = benchmarks_val_Y
        val_weights = benchmarks_weights_val

        test_X = test_set.get('benchmarks_X')
        test_Y = test_set.get('benchmarks_Y')
        test_weights = test_set.get('benchmarks_weights')
        #
        test_MSE_X = test_set.get('X').T
        test_MSE_Y = test_set.get('Y').T

        nunmber_of_features = X_train[0].shape[0]

        if dnl_batch_size is None:
            dnl_batch_size = int(len(benchmarks_train_X))
        layer_params_relu = [nunmber_of_features, hidden_layer_neurons, 1]
        mypool = mp.Pool(processes=8)

        baseline_model = relu_ppo(batch_size=mini_batch_size, max_step_size_magnitude=max_step_size_magnitude,
                                  min_step_size_magnitude=min_step_size_magnitude,
                                  layer_params=layer_params_relu, dnl_epoch=dnl_epoch,
                                  opt_params=opt_params, dnl_batch_size=dnl_batch_size,
                                  dnl_learning_rate=dnl_learning_rate,
                                  is_parallel=True, is_update_bias=is_update_bias, L2_lambda=L2_Lambda,
                                  sampling_method=DIVIDE_AND_CONQUER_GREEDY, run_time_limit=time_limit)
        baseline_model.fit_nn(X_train, Y_train, max_epochs=20)

        benchmark_x = benchmarks_train_X[benchmark_no]
        benchmark_y = benchmarks_train_Y[benchmark_no]
        benchmark_weights = benchmarks_weights_train[benchmark_no]

        for index in parameter_index:
            tovs,povs,sample_points = map_parameters_one_problem_set(baseline_model, index, opt_params, benchmark_x, benchmark_y, benchmark_weights=benchmark_weights)
            plt.contourf(sample_points[0], sample_points[1], tovs, 100)
            plt.colorbar()
            plt.xlabel("param 1")
            plt.ylabel("param 2")
            plt.savefig('figs/parameter_mapping/' + str(index) + ".pdf")
            plt.clf()
            # plt.show()
        mypool.close()
        # ax = plt.axes(projection='3d')

        # ax.contour3D(sample_points[0],sample_points[1] , tovs,50, cmap='binary')



if __name__ == "__main__":
    first_param_index = (1,(4,5))
    second_param_index = (2, (0,4))
    parameter_index = [[(1,(4,0)), (2, (0,4))], [(1,(4,1)), (2, (0,4))], [(1,(4,2)), (2, (0,4))], [(1,(4,3)), (2, (0,4))], [(1,(4,4)), (2, (0,4))], [(1,(4,5)), (2, (0,4))],
                       [(1,(4,6)), (2, (0,4))], [(1,(4,7)), (2, (0,4))], [(1,(4,8)), (2, (0,4))]]
    opt_params = get_opt_params_knapsack(capacity=12)
    map_parameters(hidden_layer_neurons=5, parameter_index=parameter_index, opt_params = opt_params)