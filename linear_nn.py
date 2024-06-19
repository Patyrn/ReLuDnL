import copy
import time

import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp

from sklearn import preprocessing

# from dnl import Sampling_Methods
# from dnl.DnlNeuralLayer import get_dnl_exact_grad
# from dnl.EnergyDataUtil import get_energy_data
# from dnl.KnapsackSolver import get_opt_params_knapsack
# from dnl.PredictPlusOptimize import PredictPlusOptModel
# from dnl.Sampling_Methods import DIVIDE_AND_CONQUER_GREEDY
# from dnl.Solver import get_optimization_objective, get_optimal_average_objective_value
# from dnl.Utils import get_train_test_split

import matplotlib.pyplot as plt





class linear_regression(nn.Module):
    def __init__(self, N, layer_params, benchmark_size=48, max_epoch=10, learning_rate=1e-2, opt_params=5,
                 dnl_learning_rate=0.1, leaky_slope=0.1, pool=None):
        """
        :param N: Number of problem sets
        :param layer_params: neural net layer configuration
        :param benchmark_size: size of the benchmark
        :param max_epoch: Max epoch numbers
        :param learning_rate: learning_rate for nn regression
        :param opt_params: optimization parameters a.k.a constraints
        :param dnl_learning_rate: learning rate for dnl
        :param pool: pool if parallel computing
        """
        super().__init__()
        self.N = N
        self.epoch_number = max_epoch
        self.learning_rate = learning_rate
        self.benchmark_size = benchmark_size
        self.opt_params = opt_params
        self.dnl_learning_rate = dnl_learning_rate
        self.scaler = None
        self.layers = nn.ModuleList()
        self.init_layers(layer_params)
        self.pool = pool
        self.leaky_slope = leaky_slope
        self.init_layers(layer_params)

    def init_layers(self, layer_params):
        self.fc1 = nn.Linear(in_features=layer_params[0], out_features=layer_params[1])

    def forward(self, x):
        x = torch.from_numpy(self.scaler.transform(x)).float()
        x = self.fc1(x)
        return x

    def fit_nn(self, x, y, max_epochs= 50):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        if self.scaler == None:
            self.scaler = preprocessing.StandardScaler().fit(x)
        y = torch.from_numpy(y).float()
        for t in range(max_epochs):
            permutation = torch.randperm(x.shape[0])
            for i in range(0, x.shape[0], self.N):

                indices = permutation[i:i + self.N]
                batch_x, batch_y = x[indices], y[indices]
                y_pred = self.forward(batch_x)
                loss = torch.sqrt(criterion(batch_y, y_pred))

                # Use autograd to compute the backward pass.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % (64 * 200) == 0:
                    print('e:{} i:{}/{} loss:{}'.format(t, i, x.size, loss.item()))
                # # Update weights using gradient descent
                # with torch.no_grad():
                #     self.w1 -= self.learning_rate * self.w1.grad
                #     self.w2 -= self.learning_rate * self.w2.grad
                #
                #     # Manually zero the gradients after updating weights
                #     self.w1.grad.zero_()
                #     self.w2.grad.zero_()
                #
        if t % 100 == 99:
            # print('y',y,'y_pred',y_pred)
            print(t, loss.item())

    def fit_dnl(self, x, y, weights):
        if self.scaler is None:
            x = torch.from_numpy(x).float()
        else:
            x = torch.from_numpy(self.scaler.transform(x)).float()

        y = torch.from_numpy(y).float()
        weights = torch.from_numpy(weights).float()

        number_of_benchmarks = int(x.size()[0] / self.benchmark_size)
        number_of_batches = int(number_of_benchmarks / self.N)

        for t in range(self.epoch_number):
            # UNCOMMENT BELOW FOR RANDOM SHUFFLE BATCHES
            # permutations = torch.randperm(number_of_benchmarks).tolist()
            # bELOW IS FOR ORDERED BATCHES
            permutations = [x for x in range(number_of_batches)]
            for i in range(0, number_of_batches, self.N):
                indices = [x for y in range(permutations[i], permutations[i] + self.N) for x in
                           range(y * self.benchmark_size, (y + 1) * self.benchmark_size)]
                batch_x, batch_y, batch_weights = x[indices], y[indices], weights[indices]

                y_pred, penultimate_x, model_params = self.forward(batch_x)
                dnl_direction = find_dnl_direction(penultimate_x, batch_y, batch_weights, model_params,
                                                   self.benchmark_size, self.opt_params, self.learning_rate,
                                                   mypool=self.pool)

                self.dnl_step(dnl_direction)

    def get_MSE(self, x, y):
        y = torch.from_numpy(y).float()
        y_pred = self.forward(x)
        loss = (y - y_pred).detach().numpy() ** 2
        return np.mean(loss)
        # if t % 100 == 99:
        #     # print('y',y,'y_pred',y_pred)
        #     print(t, loss.item())
