import torch
import torch.nn as nn
import numpy as np

from functools import partial
from sklearn import preprocessing

from Solver import get_optimization_objective, get_optimal_average_objective_value


class relu_twolayer(nn.Module):
    def __init__(self, batch_size, dnl_batch_size, layer_params, benchmark_size=48, max_epoch=10, learning_rate=1e-2,
                 dnl_epoch=10,
                 opt_params=5,
                 dnl_learning_rate=0.1, leaky_slope=0.2,
                 max_step_size_magnitude=1,
                 min_step_size_magnitude=-1,
                 verbose=False, is_Val=True, run_time_limit=10000,
                 is_parallel=False, pool=None, is_update_bias=True, L2_lambda=0.001):
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
        self.dnl_batch_size = dnl_batch_size
        self.batch_size = batch_size
        self.epoch_number = max_epoch
        self.learning_rate = learning_rate
        self.benchmark_size = benchmark_size
        self.opt_params = opt_params
        self.dnl_learning_rate = dnl_learning_rate
        self.scaler = None
        self.layers = nn.ModuleList()
        self.leaky_slope = leaky_slope
        self.init_layers(layer_params)
        self.pool = pool
        self.init_layers(layer_params)
        self.is_parallel = is_parallel
        self.dnl_epoch = dnl_epoch
        self.max_step_size_magnitude = max_step_size_magnitude
        self.min_step_size_magnitude = min_step_size_magnitude
        self.L2_lambda = L2_lambda

        self.test_run_time = 0
        self.verbose = verbose
        self.is_update_bias = is_update_bias

        self.run_time_limit = run_time_limit
        self.training_obj_value = []
        self.test_regrets = []
        self.val_regrets = []
        self.epochs = []
        self.sub_epochs = []
        self.run_time = []
        self.test_pred_y = []

    def init_layers(self, layer_params):
        self.fc1 = nn.Linear(in_features=layer_params[0], out_features=layer_params[1])
        self.fc2 = nn.Linear(in_features=layer_params[1], out_features=layer_params[2])
        # self.fc3 = nn.Linear(in_features=layer_params[2], out_features=layer_params[3])
        # self.fc4 = nn.Linear(in_features=layer_params[3], out_features=layer_params[4])
        self.relu = nn.LeakyReLU(negative_slope=self.leaky_slope)

    def forward(self, x):
        if self.scaler is not None:
            x = torch.from_numpy(self.scaler.transform(x)).float()
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        else:
            x = x.float()
        a1 = self.relu(self.fc1(x))
        # a2 = self.relu(self.fc2(a1))
        # a3 = self.relu(self.fc3(a2))
        y_pred = self.fc2(a1)
        return y_pred

    def fit_nn(self, x, y, max_epochs=None):
        if max_epochs is None:
            max_epochs = 10

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        if self.scaler == None:
            self.scaler = preprocessing.StandardScaler().fit(x)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        for t in range(max_epochs):
            permutation = torch.randperm(x.size()[0])
            for i in range(0, x.size()[0], self.batch_size):

                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = x[indices], y[indices]

                y_pred = self.forward(batch_x)
                loss = torch.sqrt(criterion(batch_y, y_pred))

                # Use autograd to compute the backward pass.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % (64 * 200) == 0:
                    print('e:{} i:{}/{} loss:{}'.format(t, i, x.size()[0], loss.item()))
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

    def test_mse(self, x, y):
        y = torch.from_numpy(y).float()
        y_pred = self.forward(x)
        loss = (y - y_pred).detach().numpy() ** 2
        return np.mean(loss)
        # if t % 100 == 99:
        #     # print('y',y,'y_pred',y_pred)
        #     print(t, loss.item())

    def get_regret(self, X, Y, weights, opt_params, pool=None):
        pred_Ys = []
        for x in X:
            pred_Ys.append(self.forward(x).detach().numpy().flatten())

        if pool is None:
            average_objective_value_with_predicted_items = get_optimization_objective(Y=Y, pred_Y=pred_Ys,
                                                                                      weights=weights,
                                                                                      opt_params=opt_params,
                                                                                      )
            optimal_average_objective_value = get_optimal_average_objective_value(Y=Y, weights=weights,
                                                                                  opt_params=self.opt_params,
                                                                                  )
            regret = np.median(optimal_average_objective_value - average_objective_value_with_predicted_items)
        else:
            map_func = partial(get_regret_worker, opt_params=self.opt_params)
            iter = zip(Y, pred_Ys, weights)
            objective_values = pool.starmap(map_func, iter)
            objective_values_predicted_items, optimal_objective_values = zip(*objective_values)
            regret = np.median(
                np.concatenate(optimal_objective_values) - np.concatenate(objective_values_predicted_items))
        return regret

    def get_objective_value(self, X, Y, weights, opt_params, pool=None):
        pred_Ys = []
        for x in X:
            pred_Ys.append(self.forward(x).detach().numpy().flatten())

        if pool is None:
            objective_values_predicted_items = get_optimization_objective(Y=Y, pred_Y=pred_Ys,
                                                                          weights=weights,
                                                                          opt_params=opt_params,
                                                                          )
            optimal_average_objective_value = get_optimal_average_objective_value(Y=Y, weights=weights,
                                                                                  opt_params=self.opt_params,
                                                                                  )
            regret = np.median(optimal_average_objective_value - objective_values_predicted_items)
        else:
            map_func = partial(get_regret_worker, opt_params=self.opt_params)
            iter = zip(Y, pred_Ys, weights)
            objective_values = pool.starmap(map_func, iter)
            objective_values_predicted_items, optimal_objective_values = zip(*objective_values)
            regret = np.median(
                np.concatenate(optimal_objective_values) - np.concatenate(objective_values_predicted_items))
            objective_values_predicted_items = np.concatenate(objective_values_predicted_items)
        return np.median(objective_values_predicted_items)


def get_regret_worker(Y, pred_Ys, weights, opt_params):
    average_objective_value_with_predicted_items = get_optimization_objective(Y=[Y], pred_Y=[pred_Ys],
                                                                              weights=[weights],
                                                                              opt_params=opt_params,
                                                                              )
    optimal_average_objective_value = get_optimal_average_objective_value(Y=[Y], weights=[weights],
                                                                          opt_params=opt_params,
                                                                          )
    return average_objective_value_with_predicted_items, optimal_average_objective_value
