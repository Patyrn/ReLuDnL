import copy
import csv
import math
import time

import numpy as np
import torch
from sklearn import model_selection, linear_model
import matplotlib.pyplot as plt

from KnapsackSolver import get_opt_params_knapsack
from Relu_DNL.EnergyDataUtil import get_energy_data
from Sampling_Methods import find_transition_intervals, Sampler, EXHAUSTIVE, MID_TRANSITION_POINT_SELECTION, \
    DIVIDE_AND_CONQUER, EDGE_TRANSITION_POINT_SELECTION, DIVIDE_AND_CONQUER_GREEDY
from Solver import compute_objective_value_single_benchmarks, \
    get_optimization_objective, get_optimal_average_objective_value
from Relu_DNL.Utils import save_results_list, get_train_test_split
from dnl.PredictPlusOptimize import PredictPlusOptModel
from linear_nn import linear_regression
from relu_2l import relu_twolayer
from relu_dnl.IconEasySolver import get_icon_instance_params
from relu_dnl.Params import KNAPSACK, ICON_SCHEDULING_EASY
from relu_dnl.dnl import Sampling_Methods
from relu_ppo import relu_ppo

np.random.seed(42)

def init_two_weight_set(benchmarks_size=50, benchmark_number=100):
    weight_seed = [1]
    benchmark_weights = generate_uniform_weights_from_seed(benchmarks_size, weight_seed).flatten().tolist()

    weights = np.array(benchmark_weights * benchmark_number).reshape(1, -1)

    sample_size = benchmark_number * benchmarks_size
    capacities = [5 if i > benchmarks_size / 2 else 7 for i in range(benchmarks_size)]
    y = np.zeros(sample_size)
    x1 = np.zeros(sample_size)
    x2 = np.zeros(sample_size)
    for i in range(benchmark_number):
        benchmarks_y = np.zeros(benchmarks_size)
        benchmarks_x1 = np.random.rand(benchmarks_size)
        for j, (c, benchmark_x1) in enumerate(zip(capacities, benchmarks_x1)):
            if c == 5:
                benchmarks_y[j] = benchmark_x1 * 5
            else:
                benchmarks_y[j] = benchmark_x1 * - 5 + 5
        y[i * benchmarks_size:(i + 1) * benchmarks_size] = benchmarks_y
        x1[i * benchmarks_size:(i + 1) * benchmarks_size] = benchmarks_x1
        x2[i * benchmarks_size:(i + 1) * benchmarks_size] = capacities
    x = np.vstack((x1, x2))
    dataset = {"x": x,
               "y": y.reshape(1, -1),
               "weights": weights}
    # ind_5 = np.where(x2==5)
    # ind_7 = np.where(x2==7)
    # plt.scatter(x1[ind_5],y[ind_5],c='r')
    #
    # plt.scatter(x1[ind_7], y[ind_7], c='b')
    # plt.show()

    return dataset


def init_bipolar_sin_set(benchmarks_size=50, benchmark_number=100):
    sample_size = benchmark_number * benchmarks_size
    b1 = 2.5
    b2 = -2
    b3 = 1.3
    b4 = 3
    x1 = np.random.rand(1, sample_size)
    c1 = np.random.rand(1, sample_size)
    # x2 = np.random.rand(1, sample_size) * + 3/10
    # x3 = np.random.rand(1, sample_size)  - 2.5/10
    # x = np.vstack((x1,x2))
    x = x1
    y = np.zeros((1, sample_size))

    noise = np.random.normal(0, 0.1, sample_size)
    # y = b1 * x1 - b3 * (x2 **2) + b2 * (x2**3)
    # y = b1 * x1 + b2 * (x1**2) + b3 * x2 + b3 * (x2**2)

    for i in range(sample_size):
        if x1[0, i] > 0.315:
            y[0, i] = np.sin(10 * x1[0, i]) * 40 + c1[0, i] + noise[i]
        else:
            y[0, i] = np.sin(10 * x1[0, i]) * 10 + c1[0, i] + noise[i]

    # plt.scatter(x1[0,:],y)
    # plt.title('Generated Data')
    # plt.xlabel("x1")
    # plt.ylabel("y")
    # plt.show()
    # benchmarks_y = np.array([np.sort(y[0,i*benchmarks_size:(i+1)*benchmarks_size]) for i in range(benchmark_number)])
    # plt.figure()
    # plt.title("Benchmark Distributions")
    # for i in range(benchmark_number):
    #     plt.subplot(10,10,i+1)
    #     plt.title(i)
    #     plt.scatter(range(benchmarks_size), benchmarks_y[i])
    #     plt.xlabel('x1')
    #     plt.ylabel('y')
    #
    # plt.show()
    dataset = {"x": x,
               "y": y}
    return dataset

def gaussian(x, alpha, r):
      return 1./(math.sqrt(alpha**math.pi))*np.exp(-alpha*np.power((x - r), 2.))

def init_norm(benchmarks_size=50, benchmark_number=100):
    weight_seed = [1]
    benchmark_weights = generate_uniform_weights_from_seed(benchmarks_size, weight_seed).flatten().tolist()

    weights = np.array(benchmark_weights * benchmark_number).reshape(1, -1).astype(int)
    sample_size = benchmark_number * benchmarks_size
    b1 = 2.5
    b2 = -2
    b3 = 1.3
    b4 = 3
    x1 = np.random.rand(1, sample_size) *10
    c1 = np.random.normal(0.5, 0.2, sample_size)
    # x2 = np.random.rand(1, sample_size) * + 3/10
    # x3 = np.random.rand(1, sample_size)  - 2.5/10
    x = np.vstack((x1, weights.flatten()))
    # x = x1.reshape(-1, 1)

    noise = np.random.normal(0, 0.1, sample_size)
    # y = b1 * x1 - b3 * (x2 **2) + b2 * (x2**3)
    # y = b1 * x1 + b2 * (x1**2) + b3 * x2 + b3 * (x2**2)
    # y = (2 ** (x1 + noise) * weights.flatten()).reshape(1, -1)
    y = 40*(gaussian(x1 + noise, 5, 5) * weights.flatten()).reshape(1, -1)
    # ax = plt.axes(projection="3d")
    # ax.scatter(x1,y)
    # plt.scatter(x1.flatten(), y.flatten())
    # plt.title('Generated Data')
    # plt.xlabel("x1")
    # plt.ylabel("y")
    # plt.show(block=False)

    benchmarks_y = np.array(
        [np.sort(y[0, i * benchmarks_size:(i + 1) * benchmarks_size]) for i in range(benchmark_number)])
    # plt.figure()
    # plt.title("Benchmark Distributions")
    # for i in range(benchmark_number):
    #     plt.subplot(10, 10, i + 1)
    #     plt.title(i)
    #     plt.scatter(range(benchmarks_size), benchmarks_y[i])
    #     plt.xlabel('x1')
    #     plt.ylabel('y')

    # plt.show()
    dataset = {"x": x,
               "y": y,
               "weights": weights.T}
    return dataset

def init_sin_set(benchmarks_size=50, benchmark_number=100):
    sample_size = benchmark_number * benchmarks_size
    b1 = 2.5
    b2 = -2
    b3 = 1.3
    b4 = 3
    x1 = np.random.normal(0.5,0.2, sample_size)
    c1 = np.random.normal(0.5,0.2, sample_size)
    # x2 = np.random.rand(1, sample_size) * + 3/10
    # x3 = np.random.rand(1, sample_size)  - 2.5/10
    # x = np.vstack((x1,x2))
    x = x1

    noise = np.random.normal(0, 2, sample_size)
    # y = b1 * x1 - b3 * (x2 **2) + b2 * (x2**3)
    # y = b1 * x1 + b2 * (x1**2) + b3 * x2 + b3 * (x2**2)
    y = np.sin(10 * x1) * 40 + c1 + noise
    # ax = plt.axes(projection="3d")
    # ax.scatter(x1,y)
    plt.scatter(x1,y)
    plt.title('Generated Data')
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.show()
    benchmarks_y = np.array([np.sort(y[0,i*benchmarks_size:(i+1)*benchmarks_size]) for i in range(benchmark_number)])
    plt.figure()
    plt.title("Benchmark Distributions")
    for i in range(benchmark_number):
        plt.subplot(10,10,i+1)
        plt.title(i)
        plt.scatter(range(benchmarks_size), benchmarks_y[i])
        plt.xlabel('x1')
        plt.ylabel('y')

    plt.show()
    dataset = {"x": x,
               "y": y}
    return dataset

def generate_uniform_weights_from_seed(benchmark_size, weight_seed):
    number_of_each_weight = int(benchmark_size / len(weight_seed))
    uniform_weights_from_seed = np.array(
        [np.ones((number_of_each_weight)) * weight for weight in weight_seed]).flatten()
    np.random.shuffle(uniform_weights_from_seed)
    uniform_weights_from_seed = uniform_weights_from_seed.reshape((1, uniform_weights_from_seed.size))

    return uniform_weights_from_seed
def init_lin_set(benchmarks_size=50, benchmark_number=100):
    weight_seed = [3,3,10, 10, 20]
    benchmark_weights = generate_uniform_weights_from_seed(benchmarks_size, weight_seed).flatten().tolist()

    weights = np.array(benchmark_weights * benchmark_number).reshape(1, -1).astype(int)
    sample_size = benchmark_number * benchmarks_size
    b1 = 2.5
    b2 = -2
    b3 = 1.3
    b4 = 3
    x1 = np.random.normal(3, 1, sample_size)
    c1 = np.random.normal(0.5, 0.2, sample_size)
    # x2 = np.random.rand(1, sample_size) * + 3/10
    # x3 = np.random.rand(1, sample_size)  - 2.5/10
    x = np.vstack((x1,weights.flatten()))
    # x = x1.reshape(-1, 1)

    noise = np.random.normal(0, 0.1, sample_size)
    # y = b1 * x1 - b3 * (x2 **2) + b2 * (x2**3)
    # y = b1 * x1 + b2 * (x1**2) + b3 * x2 + b3 * (x2**2)
    y = (2**(x1 + noise) * weights.flatten()).reshape(1, -1)
    # ax = plt.axes(projection="3d")
    # ax.scatter(x1,y)
    # plt.scatter(x1.flatten(), y.flatten())
    # plt.title('Generated Data')
    # plt.xlabel("x1")
    # plt.ylabel("y")
    # plt.show(block=False)

    benchmarks_y = np.array(
        [np.sort(y[0, i * benchmarks_size:(i + 1) * benchmarks_size]) for i in range(benchmark_number)])
    # plt.figure()
    # plt.title("Benchmark Distributions")
    # for i in range(benchmark_number):
    #     plt.subplot(10, 10, i + 1)
    #     plt.title(i)
    #     plt.scatter(range(benchmarks_size), benchmarks_y[i])
    #     plt.xlabel('x1')
    #     plt.ylabel('y')

    # plt.show()
    dataset = {"x": x,
               "y": y,
               "weights": weights.T}
    return dataset

def init_sin_set2(benchmarks_size=50, benchmark_number=100):
    sample_size = benchmark_number * benchmarks_size
    b1 = 2.5
    b2 = -2
    b3 = 1.3
    b4 = 3
    # x1 = np.random.rand(1, sample_size)
    c1 = np.random.rand(1, sample_size)
    x1 = np.zeros((1,sample_size))
    # x2 = np.random.rand(1, sample_size) * + 3/10
    # x3 = np.random.rand(1, sample_size)  - 2.5/10
    # x = np.vstack((x1,x2))
    for i in range(benchmark_number):
        x1[0,i*benchmarks_size:(i+1)*benchmarks_size] = np.linspace(0,1,num=benchmarks_size) + np.random.normal(0, 0.005)
    x = x1

    # noise = np.random.normal(0, 0.1, sample_size)
    # y = b1 * x1 - b3 * (x2 **2) + b2 * (x2**3)
    # y = b1 * x1 + b2 * (x1**2) + b3 * x2 + b3 * (x2**2)

    y = np.sin(10 * x1) * 40 + c1
    # ax = plt.axes(projection="3d")
    # ax.scatter(x1,y)
    # plt.scatter(x1,y)
    # plt.title('Generated Data')
    # plt.xlabel("x1")
    # plt.ylabel("y")
    # plt.show()
    # benchmarks_y = np.array([np.sort(y[0,i*benchmarks_size:(i+1)*benchmarks_size]) for i in range(benchmark_number)])
    # plt.figure()
    # plt.title("Benchmark Distributions")
    # for i in range(benchmark_number):
    #     plt.subplot(10,10,i+1)
    #     plt.title(i)
    #     plt.scatter(range(benchmarks_size), benchmarks_y[i])
    #     plt.xlabel('x1')
    #     plt.ylabel('y')
    #
    # plt.show()
    dataset = {"x": x,
               "y": y}
    return dataset

def init_spares_sin(benchmarks_size=50, benchmark_number=100):

    sample_size = benchmark_number * benchmarks_size
    b1 = 2.5
    b2 = -2
    b3 = 1.3
    b4 = 3
    # x1 = np.random.rand(1, sample_size)
    c1 = np.random.rand(1, sample_size)
    x1 = np.zeros((1,sample_size))
    # x2 = np.random.rand(1, sample_size) * + 3/10
    # x3 = np.random.rand(1, sample_size)  - 2.5/10
    # x = np.vstack((x1,x2))
    num = int(benchmarks_size/10)
    for i in range(benchmark_number):
        x1[0,i*benchmarks_size:(i+1)*benchmarks_size] = \
            np.hstack((np.linspace(0,0.3,num=num*1) + np.random.normal(0, 0.015),
                      np.linspace(0.3, 0.6, num=num * 8) + np.random.normal(0, 0.015),
                      np.linspace(0.6, 0.9, num=num * 1) + np.random.normal(0, 0.015)))
    x = x1

    # noise = np.random.normal(0, 0.1, sample_size)
    # y = b1 * x1 - b3 * (x2 **2) + b2 * (x2**3)
    # y = b1 * x1 + b2 * (x1**2) + b3 * x2 + b3 * (x2**2)

    y = np.sin(10 * x1) * 40 + c1
    # ax = plt.axes(projection="3d")
    # ax.scatter(x1,y)
    # plt.scatter(x1,y)
    # plt.title('Generated Data')
    # plt.xlabel("x1")
    # plt.ylabel("y")
    # plt.show()
    # benchmarks_y = np.array([np.sort(y[0,i*benchmarks_size:(i+1)*benchmarks_size]) for i in range(benchmark_number)])
    # plt.figure()
    # plt.title("Benchmark Distributions")
    # for i in range(benchmark_number):
    #     plt.subplot(10,10,i+1)
    #     plt.title(i)
    #     plt.scatter(range(benchmarks_size), benchmarks_y[i])
    #     plt.xlabel('x1')
    #     plt.ylabel('y')
    #
    # plt.show()
    dataset = {"x": x,
               "y": y}
    return dataset

def init_polynomial_set(benchmarks_size=50, benchmark_number=100):
    sample_size = benchmark_number * benchmarks_size
    b1 = 3
    b2 = 1
    b3 = 1.3
    b4 = 3
    x1 = np.random.rand(1, sample_size) * 2 - 1
    c1 = np.random.rand(1, sample_size)
    x2 = np.random.rand(1, sample_size)
    # x3 = np.random.rand(1, sample_size)  - 2.5/10
    # x = np.vstack((x1,x2))
    x = x1

    # noise = np.random.normal(0, 0.1, sample_size)
    noise = 0
    # y = b1 * x1 - b3 * (x2 **2)  + c1 + noise
    y = b1 * (x1) ** 2
    # y = b1 * x1 + b2 * (x1**2) + b3 * x2 + b3 * (x2**2)
    # y = np.sin(10 * x1) + c1 + noise
    # ax = plt.axes(projection="3d")
    # ax.scatter(x1,y)
    # plt.scatter(x1,y)
    # plt.title('Generated Data')
    # plt.xlabel("x1")
    # plt.ylabel("y")
    # plt.show()
    # benchmarks_y = np.array([np.sort(y[0,i*benchmarks_size:(i+1)*benchmarks_size]) for i in range(benchmark_number)])
    # plt.figure()
    # plt.title("Benchmark Distributions")
    # for i in range(benchmark_number):
    #     plt.subplot(10,10,i+1)
    #     plt.title(i)
    #     plt.scatter(range(benchmarks_size), benchmarks_y[i])
    #     plt.xlabel('x1')
    #     plt.ylabel('y')

    # plt.show()
    dataset = {"x": x,
               "y": y}
    return dataset


def regression():
    dataset = init_sin_set()
    # dataset = init_polynomial_set()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.get('x').T, dataset.get('y').T,
                                                                        test_size=0.2, shuffle=True)
    # print(X_train, X_test, y_train, y_test)
    layer_params_relu = [1, 100, 1]
    my_relu_regression = relu_ppo(batch_size=36, layer_params=layer_params_relu, max_epoch=10, dnl_batch_size=10)
    my_relu_regression.fit_nn(X_train, y_train, max_epochs=100)

    layer_params_linear = [1, 1]
    my_linear_regression = linear_regression(N=36, layer_params=layer_params_linear, max_epoch=100)
    my_linear_regression.fit_nn(X_train, y_train)

    scikit_regression = linear_model.Ridge().fit(X_train, y_train)

    relu_pred_y = my_relu_regression.forward(
        torch.from_numpy(X_test)).detach().numpy()
    linear_pred_y = my_linear_regression.forward(X_test).detach().numpy()
    scikit_pred = scikit_regression.predict(X_test)
    plt.figure()
    plt.scatter(X_test, relu_pred_y, c='r')
    plt.scatter(X_test, scikit_pred, c='y')
    plt.scatter(X_test, linear_pred_y, c='m')
    plt.scatter(X_test, y_test, c='b')
    plt.figure()
    plt.scatter(X_train, y_train, c='b')
    plt.show()


def split_benchmarks(X, Y, benchmark_size, weights=None):
    benchmark_number = int(len(Y) / benchmark_size)
    benchmarks_x = [[] for i in range(benchmark_number)]
    benchmarks_y = [[] for i in range(benchmark_number)]
    benchmarks_weights = [[] for i in range(benchmark_number)]
    for i in range(benchmark_number):
        benchmarks_x[i] = X[i * benchmark_size:(i + 1) * benchmark_size, :].reshape(benchmark_size, -1)
        benchmarks_y[i] = Y[i * benchmark_size:(i + 1) * benchmark_size, 0]
        if weights is not None:
            benchmarks_weights[i] = weights[i * benchmark_size:(i + 1) * benchmark_size, 0]
    return benchmarks_x, benchmarks_y, benchmarks_weights


def exhaustive_search(benchmark_x, benchmark_y, param_ind, model, layer=None):
    current_param = layer.weight[param_ind].detach().numpy()
    start_ind = current_param - abs(current_param) * 10
    end_ind = current_param + abs(current_param) * 10
    sample_params = np.array([x for x in np.linspace(start_ind, end_ind, 100)])

    weights = np.array([1 for i in range(50)])

    POVS = np.zeros(len(sample_params))
    TOVS = np.zeros(len(sample_params))
    pred_ys = np.zeros((50, 100))
    decision_policies = []
    for i, param in enumerate(sample_params):
        with torch.no_grad():
            layer.weight[param_ind] = torch.from_numpy(np.array(param)).float()
        # current_weight = temp_model.fc1.weight.detach()
        # current_weight = param
        pred_y = model.forward(torch.from_numpy(benchmark_x).float()).detach().numpy()
        pred_ys[:, i] = pred_y.T
        # tov, pov, predicted_opt_items = compute_profit_knapsack_single_benchmark(pred_y, benchmark_y,
        #                                                                          weights,
        #                                                                          capacities=[10])

        tov, pov = compute_objective_value_single_benchmarks(pred_Y=pred_y,
                                                             Y=benchmark_y,
                                                             weights=weights,
                                                             opt_params=model.opt_params)

        POVS[i] = pov
        TOVS[i] = tov

    transition_points = find_transition_intervals(alpha_samples=sample_params, predicted_profits=POVS, profits=TOVS)
    # decision_policies.append(predicted_opt_items)
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.scatter(sample_params, POVS)
    for transition_point in transition_points:
        plt.scatter((transition_point.starting_point.x + transition_point.ending_point.x) * 0.5,
                    transition_point.starting_point.predicted_profit, c='r')
    plt.title('POV')
    plt.subplot(3, 2, 2)
    plt.title('TOV')
    plt.scatter(sample_params, TOVS)
    for transition_point in transition_points:
        plt.scatter((transition_point.starting_point.x + transition_point.ending_point.x) * 0.5,
                    transition_point.starting_point.true_profit, c='r')
    plt.subplot(3, 2, 3)
    plt.title('Derivatives')
    shifted_POVS = np.roll(POVS, 1)
    der = (POVS[1:99] - shifted_POVS[1:99]) / (sample_params[1] - sample_params[2])
    plt.scatter(sample_params[1:99], der)
    shifted_der = np.roll(der, 1)
    d2 = (der[2:98] - shifted_der[2:98]) / (sample_params[1] - sample_params[2])

    plt.subplot(3, 2, 4)
    plt.title('Second Derivatives')
    plt.scatter(sample_params[2:98], d2)
    # for i in range(96):
    #     print("decision policy({}): {}, der2: {}".format(i, decision_policies[i + 2], np.round(d2[i], 3)))

    plt.subplot(3, 2, 5)
    plt.title('Individual Item Predictions')
    for i in range(50):
        plt.plot(sample_params, pred_ys[i, :])
    plt.subplot(3, 2, 6)
    plt.title("MSE")
    mse = []
    for i in range(100):
        mse.append(np.mean((benchmark_y - pred_ys[:, i]) ** 2))
    plt.scatter(sample_params, mse)

    # plt.figure()
    # for i in range(100):
    #     plt.title("pred_values vs real values")
    #     plt.subplot(10,10,i+1)
    #     plt.scatter(np.array([i for i in range(50)]), pred_ys[:,i], c='r')
    #     plt.scatter(np.array([i for i in range(50)]), benchmark_y, c='b')
    plt.show()
    return POVS, TOVS


def divide_and_conquer_search(benchmark_x, benchmark_y, param_ind, model, layer=None):
    current_param = layer.weight[param_ind].detach().numpy()
    start_ind = current_param - abs(current_param) * 10
    end_ind = current_param + abs(current_param) * 10
    sample_params = np.array([x for x in np.linspace(start_ind, end_ind, 100)])

    weights = np.array([1 for i in range(50)])

    POVS = np.zeros(len(sample_params))
    TOVS = np.zeros(len(sample_params))
    pred_ys = np.zeros((50, 100))

    # M =
    decision_policies = []
    for i, param in enumerate(sample_params):
        with torch.no_grad():
            layer.weight[param_ind] = torch.from_numpy(np.array(param)).float()
        # current_weight = temp_model.fc1.weight.detach()
        # current_weight = param
        pred_y = model.forward(torch.from_numpy(benchmark_x).float()).detach().numpy()
        pred_ys[:, i] = pred_y.T
        # tov, pov, predicted_opt_items = compute_profit_knapsack_single_benchmark(pred_y, benchmark_y,
        #                                                                          weights,
        #                                                                          capacities=[10])

        tov, pov = compute_objective_value_single_benchmarks(pred_Y=pred_y,
                                                             Y=benchmark_y,
                                                             weights=weights,
                                                             opt_params=model.opt_params)

        POVS[i] = pov
        TOVS[i] = tov

    transition_points = find_transition_intervals(alpha_samples=sample_params, predicted_profits=POVS, profits=TOVS)
    # decision_policies.append(predicted_opt_items)
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.scatter(sample_params, POVS)
    for transition_point in transition_points:
        plt.scatter((transition_point.starting_point.x + transition_point.ending_point.x) * 0.5,
                    transition_point.starting_point.predicted_profit, c='r')
    plt.title('POV')
    plt.subplot(3, 2, 2)
    plt.title('TOV')
    plt.scatter(sample_params, TOVS)
    for transition_point in transition_points:
        plt.scatter((transition_point.starting_point.x + transition_point.ending_point.x) * 0.5,
                    transition_point.starting_point.true_profit, c='r')
    plt.subplot(3, 2, 3)
    plt.title('Derivatives')
    shifted_POVS = np.roll(POVS, 1)
    der = (POVS[1:99] - shifted_POVS[1:99]) / (sample_params[1] - sample_params[2])
    plt.scatter(sample_params[1:99], der)
    shifted_der = np.roll(der, 1)
    d2 = (der[2:98] - shifted_der[2:98]) / (sample_params[1] - sample_params[2])

    plt.subplot(3, 2, 4)
    plt.title('Second Derivatives')
    plt.scatter(sample_params[2:98], d2)
    # for i in range(96):
    #     print("decision policy({}): {}, der2: {}".format(i, decision_policies[i + 2], np.round(d2[i], 3)))

    plt.subplot(3, 2, 5)
    plt.title('Individual Item Predictions')
    for i in range(50):
        plt.plot(sample_params, pred_ys[i, :])
    plt.subplot(3, 2, 6)
    plt.title("MSE")
    mse = []
    for i in range(100):
        mse.append(np.mean((benchmark_y - pred_ys[:, i]) ** 2))
    plt.scatter(sample_params, mse)

    # plt.figure()
    # for i in range(100):
    #     plt.title("pred_values vs real values")
    #     plt.subplot(10,10,i+1)
    #     plt.scatter(np.array([i for i in range(50)]), pred_ys[:,i], c='r')
    #     plt.scatter(np.array([i for i in range(50)]), benchmark_y, c='b')
    plt.show()
    return POVS, TOVS


def transition_point_search():
    benchmark_size = 50
    benchmark_number = 100
    dataset = init_polynomial_set(benchmark_size, benchmark_number)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.get('x').T, dataset.get('y').T,
                                                                        test_size=0.2, shuffle=True)
    layer_params_relu = [1, 10, 1]
    my_relu_regression = relu_ppo(N=10, layer_params=layer_params_relu, max_epoch=20,
                                  opt_params=get_opt_params_knapsack(capacity=10))
    my_relu_regression.fit_nn(X_train, y_train)

    layer_params_linear = [1, 1]
    my_linear_regression = linear_regression(N=10, layer_params=layer_params_linear, max_epoch=20)
    my_linear_regression.fit_nn(X_train, y_train)

    relu_pred_y = my_relu_regression.forward(
        torch.from_numpy(my_relu_regression.scaler.transform(X_test)).float()).detach().numpy()
    linear_pred_y = my_linear_regression.forward(
        torch.from_numpy(my_linear_regression.scaler.transform(X_test)).float()).detach().numpy()

    benchmarks_x, benchmarks_y = split_benchmarks(X_test, y_test, benchmark_size, benchmark_number)
    test_benchmark_no = 0
    benchmark_x = benchmarks_x[test_benchmark_no]
    benchmark_y = benchmarks_y[test_benchmark_no]

    layer_no = 0
    param_no = 5
    current_param = my_relu_regression.fc1.weight[param_no].detach().numpy()
    temp_model = copy.deepcopy(my_relu_regression)
    max_step_size_magnitude = 0
    min_step_size_magnitude = -1
    exhaustive_sampler = Sampler(sampling_method=EXHAUSTIVE, max_step_size_magnitude=max_step_size_magnitude,
                                 min_step_size_magnitude=min_step_size_magnitude,
                                 transition_point_selection=MID_TRANSITION_POINT_SELECTION)
    dnc_sampler = Sampler(sampling_method=DIVIDE_AND_CONQUER, max_step_size_magnitude=max_step_size_magnitude,
                          min_step_size_magnitude=min_step_size_magnitude,
                          transition_point_selection=EDGE_TRANSITION_POINT_SELECTION)
    weights = np.array([1 for i in range(50)])

    for layer_no in range(2):
        if layer_no == 0:
            current_temp_layer = temp_model.fc1
            current_original_layer = my_relu_regression.fc1



        else:
            current_temp_layer = temp_model.fc2
            current_original_layer = my_relu_regression.fc2

        prev_ind = None
        for param_ind in [(i, j) for i in range(current_original_layer.weight.size()[0]) for j in
                          range(current_original_layer.weight.size()[1])]:

            # Check for efficiency copying. Find a way to impelemnt divide and conquer without changing original layers

            # print("exh layer: {}, param_ind: {}".format(layer_no, param_ind))
            exh_start_time = time.time()
            exh_points, exh_intervals, sample_params, exh_POVS, exh_TOVS = exhaustive_sampler.get_transition_points(
                temp_model, current_temp_layer, benchmark_x, benchmark_y, weights, param_ind)
            with torch.no_grad():
                current_temp_layer.weight[param_ind] = current_original_layer.weight[param_ind]

            exh_end_time = time.time()
            print("EXH finished in {}s".format(exh_end_time - exh_start_time))
            # print("dnc layer: {}, param_ind: {}".format(layer_no, param_ind))
            dnc_start_time = time.time()
            dnc_points, dnc_intervals, __, dnc_POVS, dnc_TOVS = dnc_sampler.get_transition_points(temp_model,
                                                                                                  current_temp_layer,
                                                                                                  benchmark_x,
                                                                                                  benchmark_y, weights,
                                                                                                  param_ind)
            dnc_end_time = time.time()
            print("DNC finished in {}s".format(dnc_end_time - dnc_start_time))

            with torch.no_grad():
                current_temp_layer.weight[param_ind] = current_original_layer.weight[param_ind]

            plt.subplot(2, 1, 1)
            plt.scatter(sample_params, exh_POVS)
            for point in dnc_points:
                plt.scatter(point.x, point.predicted_profit, c='r')

            plt.subplot(2, 1, 2)
            plt.scatter(sample_params, exh_TOVS)
            for point in dnc_points:
                plt.scatter(point.x, point.true_profit, c='r')

            # exhaustive_search(benchmark_x=benchmark_x, benchmark_y=benchmark_y, param_ind=param_ind, model=temp_model,
            #                   layer=current_temp_layer)
            plt.show()


def fit_dnl(divider=1, capacity = 20):
    dnl_regression_epoch = 5
    regression_epoch = 5
    is_update_bias = True
    col_range = 2
    row_range = 3
    dnl_epoch = col_range*row_range
    capacity = capacity
    # dnl_epoch = 10

    benchmark_size = 50
    benchmark_numbers = 300
    dnl_batch_size = benchmark_numbers
    # dataset = init_polynomial_set(benchmark_size, benchmark_numbers)
    # dataset = init_sin_set2(benchmark_size, benchmark_numbers)
    # dataset = init_spares_sin(benchmark_size, benchmark_numbers)
    # dataset = init_bipolar_sin_set(benchmark_size, benchmark_numbers)
    # dataset = init_lin_set(benchmark_size, benchmark_numbers)
    dataset = init_norm(benchmark_size, benchmark_numbers)
    # dataset = init_two_weight_set(benchmark_size, benchmark_numbers)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.get('x').T, dataset.get('y').T,
                                                                        test_size=0.3, shuffle=True)
    in_features = 2
    weights = dataset.get('weights')
    val_size = int(benchmark_numbers * 0.1)
    val_x = X_train[0:val_size * benchmark_size]
    val_y = y_train[0:val_size * benchmark_size]
    val_weights = weights[0:val_size * benchmark_size]
    layer_params_relu = [in_features, 4, 4, 4, 1]
    lin_layer_params = [in_features,1]
    layer_params_twolayer_relu = [in_features, 5, 1]
    opt_params = get_opt_params_knapsack(capacity=capacity)
    dnl_batch_size = int(len(y_train) / benchmark_size)
    my_relu_regression = relu_ppo(batch_size=32,
                                  layer_params=layer_params_relu, max_epoch=10,
                                  opt_params=opt_params, dnl_batch_size=dnl_batch_size, is_parallel=True)
    my_relu_dnl = relu_ppo(batch_size=32, max_step_size_magnitude=2,
                           min_step_size_magnitude=-1,
                           layer_params=layer_params_relu, dnl_epoch=dnl_epoch,
                           opt_params=opt_params, dnl_batch_size=-1,
                           dnl_learning_rate=1, params_per_epoch_divider= divider,
                           is_parallel=True, is_update_bias=is_update_bias, L2_lambda=0.001,
                           sampling_method=DIVIDE_AND_CONQUER_GREEDY, run_time_limit=3000)
    my_lin_dnl = relu_ppo(batch_size=32, max_step_size_magnitude=2,
                           min_step_size_magnitude=-1,
                           layer_params=lin_layer_params, dnl_epoch=dnl_epoch,
                           opt_params=opt_params, dnl_batch_size=-1,
                           dnl_learning_rate=1, params_per_epoch_divider= divider,
                           is_parallel=True, is_update_bias=is_update_bias, L2_lambda=0.001,
                           sampling_method=DIVIDE_AND_CONQUER_GREEDY, run_time_limit=3000)
    my_relu_twolayer = relu_twolayer(batch_size=16,
                                     layer_params=layer_params_twolayer_relu, max_epoch=10,
                                     opt_params=opt_params, dnl_batch_size=dnl_batch_size, is_parallel=True)

    my_relu_twolayer.fit_nn(X_train, y_train, max_epochs=dnl_regression_epoch)
    my_relu_regression.fit_nn(X_train, y_train, max_epochs=regression_epoch)
    my_lin_dnl.fit_nn(X_train, y_train, max_epochs=dnl_regression_epoch)
    my_relu_dnl.fit_nn(X_train, y_train, max_epochs=dnl_regression_epoch)

    my_linear_regression = linear_regression(N=10, layer_params=lin_layer_params, max_epoch=regression_epoch)
    my_linear_regression.fit_nn(X_train, y_train, max_epochs=regression_epoch)

    benchmarks_x, benchmarks_y, benchmarks_weights = split_benchmarks(X_train, y_train, benchmark_size, weights=weights)
    benchmarks_x_test, benchmarks_y_test, benchmarks_weights_test = split_benchmarks(X_test, y_test, benchmark_size,
                                                                                     weights=weights)
    benchmarks_x_val, benchmarks_y_val, benchmarks_weights_val = split_benchmarks(val_x, val_y, benchmark_size,
                                                                                  weights=val_weights)

    linear_regression_regret = get_regret(my_linear_regression, benchmarks_x_test, benchmarks_y_test,
                                          benchmarks_weights_test, opt_params)
    my_relu_regression_regret = get_regret(my_relu_regression, benchmarks_x_test, benchmarks_y_test,
                                           benchmarks_weights_test, opt_params)
    my_relu_dnl_regret = get_regret(my_relu_dnl, benchmarks_x_test, benchmarks_y_test,
                                    benchmarks_weights_test, opt_params)
    my_two_layer_regret = get_regret(my_relu_twolayer, benchmarks_x_test, benchmarks_y_test,
                                     benchmarks_weights_test, opt_params)

    my_lin_dnl_regret = get_regret(my_lin_dnl, benchmarks_x_test, benchmarks_y_test,
                                    benchmarks_weights_test, opt_params)
    print("**************************************")
    print(
        "First linear regression regret: {}, NN regression regret: {}, DNL regression regret: {}, Lin Dnl Regret: {}".format(
            linear_regression_regret,
            my_relu_regression_regret, my_relu_dnl_regret, my_lin_dnl_regret))


    my_lin_dnl.fit_dnl(benchmarks_x, benchmarks_y, benchmarks_weights, benchmarks_x_val, benchmarks_y_val,
                        benchmarks_weights_val,
                        benchmark_size=benchmark_size, test_X=benchmarks_x_test, test_Y=benchmarks_y_test,
                        test_weights=benchmarks_weights_test, test_X_MSE=X_test, print_test=True)

    my_relu_dnl.fit_dnl(benchmarks_x, benchmarks_y, benchmarks_weights, benchmarks_x_val, benchmarks_y_val,
                        benchmarks_weights_val,
                        benchmark_size=benchmark_size, test_X=benchmarks_x_test, test_Y=benchmarks_y_test,
                        test_weights=benchmarks_weights_test, test_X_MSE=X_test, print_test=True)
    print('Linear DNL')


    post_my_relu_regression_regret = get_regret(my_relu_regression, benchmarks_x_test, benchmarks_y_test,
                                                benchmarks_weights_test, opt_params)
    post_my_relu_dnl_regret = get_regret(my_relu_dnl, benchmarks_x_test, benchmarks_y_test,
                                         benchmarks_weights_test, opt_params)
    post_my_lin_dnl_regret = get_regret(my_lin_dnl, benchmarks_x_test, benchmarks_y_test,
                                         benchmarks_weights_test, opt_params)

    dnl_relu_pred_y = my_relu_dnl.forward(
        torch.from_numpy(X_test)).detach().numpy()
    relu_pred_y = my_relu_regression.forward(
        torch.from_numpy(X_test)).detach().numpy()
    lin_dnl_pred_y = my_lin_dnl.forward(
        torch.from_numpy(X_test)).detach().numpy()


    linear_pred_y = my_linear_regression.forward(X_test).detach().numpy()
    two_layer_pred_y = my_relu_twolayer.forward(X_test).detach().numpy()
    save_results_list("print_results.csv", "", my_relu_dnl.print())
    plt.figure()
    # plt.scatter(X_test[:, 0], relu_pred_y, c='y')
    # plt.scatter(X_test[:, 0], linear_pred_y, c='m')
    # plt.scatter(X_test[:, 0], dnl_relu_pred_y, c='r')
    # # plt.scatter(X_test[:, 0], two_layer_pred_y, c='g')
    # plt.scatter(X_test[:, 0], y_test, c='b')
    plt.subplot(2, 2, 1)
    plt.scatter(X_test[:, 0], lin_dnl_pred_y, c='r', label="linear_pred_y")
    # plt.scatter(X_test[:, 0], linear_pred_y, c='m')
    plt.scatter(X_test[:, 0], dnl_relu_pred_y, c='y', label='Relu-DNL')
    plt.scatter(X_test[:, 0], y_test, c='b', label="data")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Capacity: {}, ReluDnl Regret: {}".format(capacity, post_my_relu_dnl_regret))
    plt.scatter(X_test[:, 0], dnl_relu_pred_y, c='y', label='Relu-DNL')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Capacity: {}, linear Regret: {}".format(capacity, linear_regression_regret))
    plt.scatter(X_test[:, 0], linear_pred_y, c='m', label="linear regression")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("Capacity: {}, Dnl Regret: {}".format(capacity,post_my_lin_dnl_regret))
    plt.scatter(X_test[:, 0], lin_dnl_pred_y, c='r', label="linear_pred_y")




    plt.legend()
    plt.title("Capacity: {}, Lin Regret: {}, ReluDnl Regret: {}, Dnl Regret: {}".format(capacity, linear_regression_regret, post_my_relu_dnl_regret, post_my_lin_dnl_regret))
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.scatter(X_train[:, 0], y_train, c='b')
    # plt.subplot(2, 1, 2)
    # plt.scatter(X_test[:, 0], dnl_relu_pred_y, c='y')
    plt.figure()

    pred_y_per_epoch = my_relu_dnl.test_pred_y

    for i in range(row_range):
        for j in range(col_range):
            plt.subplot(row_range, col_range, col_range * i + j + 1)
            plt.scatter(X_test[:, 0], pred_y_per_epoch[col_range * i + j], c='r')
            plt.scatter(X_test[:, 0], y_test, c='b')

    plt.figure()
    for i in range(row_range):
        for j in range(col_range):
            plt.subplot(row_range, col_range, col_range * i + j + 1)
            plt.scatter(X_test[:, 0], pred_y_per_epoch[col_range * i + j], c='r')
    plt.show()


def fit_dnl_two_weight():
    dnl_regression_epoch = 1
    dnl_epoch = 1
    regression_epoch = 1

    benchmark_size = 50
    benchmark_numbers = 100
    dnl_batch_size = benchmark_numbers
    # dataset = init_polynomial_set(benchmark_size, benchmark_numbers)
    dataset = init_two_weight_set(benchmark_size, benchmark_numbers)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset.get('x').T, dataset.get('y').T,
                                                                        test_size=0.2, shuffle=True)
    weights = dataset.get('weights')
    val_x = X_train[0:30]
    val_y = y_train[0:30]
    val_weights = weights[0:30]
    layer_params_relu = [2, 5, 5, 5, 1]
    opt_params = get_opt_params_knapsack(capacity=7)
    my_relu_regression = relu_ppo(batch_size=16,
                                  layer_params=layer_params_relu, max_epoch=dnl_epoch,
                                  opt_params=opt_params, dnl_batch_size=dnl_batch_size, is_parallel=True)
    my_relu_dnl = relu_ppo(batch_size=16,
                           layer_params=layer_params_relu, max_epoch=dnl_epoch,
                           opt_params=opt_params, dnl_batch_size=dnl_batch_size, dnl_learning_rate=0.01,
                           is_parallel=True)
    my_relu_regression.fit_nn(X_train, y_train, max_epochs=dnl_regression_epoch)
    my_relu_dnl.fit_nn(X_train, y_train, max_epochs=dnl_regression_epoch)

    layer_params_linear = [2, 1]
    my_linear_regression = linear_regression(N=10, layer_params=layer_params_linear, max_epoch=regression_epoch)
    my_linear_regression.fit_nn(X_train, y_train)

    benchmarks_x, benchmarks_y, benchmarks_weights = split_benchmarks(X_train, y_train, benchmark_size, weights=weights)
    benchmarks_x_test, benchmarks_y_test, benchmarks_weights_test = split_benchmarks(X_test, y_test, benchmark_size,
                                                                                     weights=weights)

    linear_regression_regret = get_regret(my_linear_regression, benchmarks_x_test, benchmarks_y_test,
                                          benchmarks_weights_test, opt_params)
    my_relu_regression_regret = get_regret(my_relu_regression, benchmarks_x_test, benchmarks_y_test,
                                           benchmarks_weights_test, opt_params)
    my_relu_dnl_regret = get_regret(my_relu_regression, benchmarks_x_test, benchmarks_y_test,
                                    benchmarks_weights_test, opt_params)
    print("**************************************")
    print("First linear regression regret: {}, NN regression regret: {}, DNL regression regret: {}".format(
        linear_regression_regret,
        my_relu_regression_regret, my_relu_dnl_regret))

    my_relu_dnl.fit_dnl(benchmarks_x, benchmarks_y, benchmarks_weights, val_x, val_y, val_weights,
                        benchmark_size=benchmark_size, test_X=benchmarks_x_test, test_Y=benchmarks_y_test,
                        test_weights=benchmarks_weights_test, test_X_MSE=X_test)

    dnl_relu_pred_y = my_relu_dnl.forward(
        torch.from_numpy(X_test)).detach().numpy()
    relu_pred_y = my_relu_regression.forward(
        torch.from_numpy(X_test)).detach().numpy()
    linear_pred_y = my_linear_regression.forward(X_test).detach().numpy()

    plt.figure()
    plt.scatter(X_test[:, 0], relu_pred_y, c='r', label="relu_regression")
    # plt.scatter(X_test[:, 0], linear_pred_y, c='m')
    plt.scatter(X_test[:, 0], dnl_relu_pred_y, c='y', label='Relu-DNL')
    plt.scatter(X_test[:, 0], y_test, c='b', label="data")
    plt.legend()
    plt.figure()
    plt.scatter(X_train[:, 0], y_train, c='b')

    # plt.figure()
    # pred_y_per_epoch = my_relu_dnl.test_pred_y

    # row_range = 5
    # col_range = 10
    # for i in range(row_range):
    #     for j in range(col_range):
    #         plt.subplot(row_range, col_range, col_range * i + j + 1)
    #         plt.scatter(X_test[:, 0], pred_y_per_epoch[col_range * i + j], c='r')
    #         plt.scatter(X_test[:, 0], y_test, c='b')
    plt.show()


def get_regret(model, X, Y, weights, opt_params):
    pred_Ys = []
    for x in X:
        pred_Ys.append(model.forward(x).detach().numpy().flatten())
    average_objective_value_with_predicted_items = get_optimization_objective(Y=Y, pred_Y=pred_Ys, weights=weights,
                                                                              opt_params=opt_params,
                                                                              )
    optimal_average_objective_value = get_optimal_average_objective_value(Y=Y, weights=weights,
                                                                          opt_params=opt_params,
                                                                          )
    regret = np.median(optimal_average_objective_value - average_objective_value_with_predicted_items)

    return regret


def linear_regression_test():
    dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=False,
                              kfold=0, noise_level=0)

    # combine weights with X first
    # may need to split weights

    train_set, test_set = get_train_test_split(dataset, random_seed=0, is_shuffle=False)

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

    dnl_model = PredictPlusOptModel()
    dnl_model.init_params_lin_regression(X=X_train, Y=Y_train)
    dnl_MSE = dnl_model.get_MSE(test_MSE_X, test_MSE_Y)

    # LINEAR REGRESSION ONE LAYER
    one_layer_regression = linear_regression(N=32, layer_params=[9, 1])
    one_layer_regression.init_layers([9, 1])
    one_layer_regression.fit_nn(X_train, Y_train, max_epochs=20)
    regression_mse = one_layer_regression.get_MSE(test_MSE_X, test_MSE_Y)
    print("DNL MSE: {}, Linear MSE: {}".format(dnl_MSE, regression_mse))
    two_layer_regression = relu_ppo(batch_size=32, dnl_batch_size=32,
                                    layer_params=[9, 5, 1],
                                    is_parallel=True,
                                    sampling_method=DIVIDE_AND_CONQUER)
    two_layer_regression.fit_nn(X_train, Y_train, max_epochs=20)
    MSE = two_layer_regression.get_MSE(test_MSE_X, test_MSE_Y)

    with open("regression_comparison.csv", 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        csvwriter.writerow(["DNL: {}".format(dnl_MSE), "1layer: {}".format(regression_mse), "2layer: {}".format(MSE)])


def get_optimization_objectives_knapsack():

    for fold in range(4):
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=False,
                                  kfold=fold, noise_level=0)

        # combine weights with X first
        # may need to split weights

        train_set, test_set = get_train_test_split(dataset, random_seed=0, is_shuffle=False)

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
        #Knapsack
        for c in [12, 24, 96, 172, 196, 220]:
            opt_params = get_opt_params_knapsack(c)
            obj_value = get_optimal_average_objective_value(test_Y, test_weights, opt_params, solver=KNAPSACK)
            str = "{} , {}".format(c,np.mean(obj_value))
            with open("knapsack_objectives.csv", 'a+', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ')
                csvwriter.writerow(str)
        #Scheduling
        for load in [30, 31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,50,51,52,5.,54,55,56,57,400,500,501]:
            opt_params = get_icon_instance_params(load)
            obj_value = get_optimal_average_objective_value(test_Y, test_weights, opt_params, solver=ICON_SCHEDULING_EASY)
            str = "{},{}".format(load, obj_value)
            with open("icon_scheduling.csv", 'a+', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ')
                csvwriter.writerow(str)
def get_optimization_objectives_scheduling():

    for fold in range(4):
        dataset = get_energy_data('energy_data.txt', generate_weight=True, unit_weight=True,
                                  kfold=fold, noise_level=0)

        # combine weights with X first
        # may need to split weights

        train_set, test_set = get_train_test_split(dataset, random_seed=0, is_shuffle=True)

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


        #Scheduling
        for load in [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,50,51,52,53,54,55,56,57,400,500,501]:
            print(load)
            opt_params = get_icon_instance_params(load)
            obj_value = get_optimal_average_objective_value(test_Y, test_weights, opt_params, solver=ICON_SCHEDULING_EASY)
            str = "{},{}".format(load, np.mean(obj_value))
            with open("icon_scheduling.csv", 'a+', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ')
                csvwriter.writerow(str)


if __name__ == "__main__":
    # regression()

    # transition_point_search()
    # init_two_weight_set()
    # init_gaussian(benchmarks_size=50, benchmark_number=100)
    # init_sin_set()`
    # init_lin_set()
    fit_dnl(divider=10,capacity=1)
    # fit_dnl_two_weight()
    # for i in range(10):
    #     linear_regression_test()
    # get_optimization_objectives_knapsack()
    # get_optimization_objectives_scheduling()