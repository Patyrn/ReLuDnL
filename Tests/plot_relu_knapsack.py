import copy
from itertools import zip_longest

import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from relu_dnl.Relu_DNL.Utils import read_file

Models = [""]
ICON_KNAPSACK = 'Iconknap'
ICON_KNAPSACK_FOLDER = 'Icon_knapsack'

ICON_SCHEDULING = "Iconsched"
ICON_SCHEDULING_FOLDER = 'Icon_scheduling'

RELU_RUN_TIME_IND = 2
RELU_VAL_REGRET_IND = 4
RELU_OBJECTIVE_IND = 5
RELU_REGRET_IND = 6
RELU_REGRET_RATIO_IND = 7
RELU_MSE_IND = 8

SPO_VAL_REGRET_IND = 0
SPO_RUN_TIME_IND = 2
SPO_REGRET_IND = 1

INTOPT_VAL_REGRET_IND = 0
INTOPT_RUN_TIME_IND = 2
INTOPT_REGRET_IND = 1

# facecolors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
facecolors = ['#a6cee3',
              '#1f78b4',
              '#b2df8a',
              '#33a02c',
              '#a6cee3',
              '#1f78b4',
              '#b2df8a',
              '#33a02c',
              '#a6cee3',
              '#000000'
              ]
# facecolors = []
# ecolors = ["#a87400", "#428ab3", "#007354", "#915777", "#696969", "#a87400", "#00456b", "#428ab3"]
ecolors = ['#a6cee3',
           '#1f78b4',
           '#b2df8a',
           '#33a02c']
# patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '+')
patterns = ('', '', '', '', '/', '--', '\\', 'o', '+', '')


def parse_file_ReLu(f_name="Iconknap_l30k0_ReLuDNL_l01.csv", f_folder=None):
    if f_folder is None:
        f_folder = "Tests/{}/{}".format(ICON_KNAPSACK_FOLDER, "ReLuDNL")
    df = read_file(filename=f_name, folder_path=f_folder)
    df = seperate_models(df)
    return df


def parse_file_spo(f_name="Load30SPOmax_spartan_kfold1.csv", f_folder=None):
    if f_folder is None:
        f_folder = "Tests/{}/{}".format(ICON_KNAPSACK_FOLDER, "SPORelax")
    df = read_file(filename=f_name, folder_path=f_folder, delimiter=',')
    return df

def parse_file_intopt(f_name="Load30SPOmax_spartan_kfold1.csv", f_folder=None):
    if f_folder is None:
        f_folder = "Tests/{}/{}".format(ICON_KNAPSACK_FOLDER, "IntOpt")
    df = read_file(filename=f_name, folder_path=f_folder, delimiter=',')
    return df


class test_data():
    def __init__(self, opt=None, l=None, p=None, k=None, data=None):
        self.opt = opt
        self.l = l
        self.p = p
        self.k = k
        self.data = data


class test_df():
    def __init__(self, param_range=None, data=None):
        self.param_range = param_range
        self.data = data


def combine_data_scheduling(file_name_list, capacities, layer_params_list, kfolds):
    df_temp = [[[[None for k in range(len(kfolds))] for p in range(3)] for
                l in range(len(layer_params_list))] for c in range(len(capacities))]

    for ci, c in enumerate(capacities):
        for li, l in enumerate(layer_params_list):
            param_range = set()
            tmp_data = [[] for ki in enumerate(kfolds)]
            for ki, k in enumerate(kfolds):
                data = parse_file_ReLu(f_name=file_name_list[ci][li][ki])
                for test_data in data:
                    test_data.opt = "c{}".format(c)
                    test_data.l = l
                    test_data.k = k
                    param_range.add(int(test_data.p))
                tmp_data[ki].extend(data)
            param_range = sorted(param_range)
            # print('layer')
            for ki, k in enumerate(kfolds):
                for data in tmp_data[ki]:
                    try:
                        df_temp[ci][li][param_range.index(int(data.p))][ki] = data
                        # print('data p: {} i: {} k: {}'.format(data.p,param_range.index(data.p), ki))
                    except:
                        print('error')
                    # df[mi][ji][li][ki].extend(data)
    return df_temp


def seperate_models(df):
    param_per_epoch_loc = 4
    epoch_loc = 0

    # param_divider_set = set()
    param_divider_list = []
    seperated_df = []
    params_per_epoch_index = None
    for r_index, row in enumerate(df):
        if row[param_per_epoch_loc] == 'params_per_epoch':
            params_per_epoch_val = df[r_index + 1][param_per_epoch_loc]
            if params_per_epoch_val in param_divider_list:
                params_per_epoch_index = param_divider_list.index(params_per_epoch_val)
            else:
                param_divider_list.append(params_per_epoch_val)
                params_per_epoch_index = param_divider_list.index(params_per_epoch_val)
                seperated_df.append(test_data(p=params_per_epoch_val, data=[]))
        elif is_float(row[epoch_loc]):
            seperated_df[params_per_epoch_index].data.append(row)

    return seperated_df


def get_spo_df(file_name_list, capacities, kfolds):
    df_temp = [[None for k in range(len(kfolds))] for m in range(len(capacities))]

    for ci, c in enumerate(capacities):
        for ki, k in enumerate(kfolds):
            data = parse_file_spo(f_name=file_name_list[ci][ki])
            data = [list(map(arr.__getitem__, [4, 7, 10])) for arr in data[1:]]
            df_temp[ci][ki] = test_data(data=data[1:], opt="c{}".format(c), k=k)
    return df_temp


def get_non_linear_spo_df(file_name_list, capacities, layer_params_list, kfolds):
    df_temp = [[[None for k in range(len(kfolds))] for
                l in range(len(layer_params_list))] for c in capacities]

    for ci, c in enumerate(capacities):
        for li, l in enumerate(layer_params_list):
            for ki, k in enumerate(kfolds):
                data = parse_file_spo(f_name=file_name_list[ci][li][ki])
                data = [list(map(arr.__getitem__, [4, 7, 10])) for arr in data[1:]]
                df_temp[ci][li][ki] = test_data(data=data[1:], opt="c{}".format(c), k=k)
    return df_temp

def get_intopt_df(file_name_list, capacities, kfolds):
    df_temp = [[None for k in range(len(kfolds))] for m in range(len(capacities))]

    for ci, c in enumerate(capacities):
        for ki, k in enumerate(kfolds):
            data = parse_file_intopt(f_name=file_name_list[ci][ki])
            data = [list(map(arr.__getitem__, [0, 2, 7])) for arr in data[1:]]
            df_temp[ci][ki] = test_data(data=data[1:], opt="c{}".format(c), k=k)
    return df_temp


def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def generate_file_name_ReLu_scheduling(prefix='Iconsched', load=30, fold=0, model="ReLuDNL", layer_params='0901'):
    f_name = "{}_l{}k{}_{}_l{}.csv".format(prefix, load, fold, model, layer_params)
    return f_name


def generate_file_name_ReLu_knapsack(prefix='Iconknap', capacity=12, fold=0, model="ReLuDNL", layer_params='0901'):
    f_name = "{}_c{}k{}_{}_l{}.csv".format(prefix, capacity, fold, model, layer_params)
    return f_name


def generate_file_name_SPO_knapsack(capacity=12, fold=0):
    f_name = "gurobi_knapsack_SPOk{}_c{}.csv".format(fold, capacity)
    return f_name

def generate_file_name_intopt_knapsack(capacity=12, fold=0):
    f_name = "0lintoptc{}k{}.csv".format(capacity, fold)
    return f_name

def generate_file_name_non_linear_SPO_knapsack(capacity=12, fold=0, layer_params='0901'):
    f_name = "Iconknap_c{}k{}_SPO_l09{}.csv".format(capacity, fold, layer_params)
    return f_name


def bar_plots_all_models(df, capacities, layer_params_list,
                         param_range, kfolds, df_spo, df_relu_spo, df_intopt, xtick_labels=None, is_normalize=False, ylim=None, is_save= True):
    plot_title = "Weighted Knapsack"

    fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ind = np.arange(0, 6)
    # ax1.set_xticks(ind)
    # ax1.set_xticklabels(capacities)
    # ax1.set_title('Run Time vs Sample Points')
    # ax1.set_xlabel('Sample Points')
    # ax1.set_ylabel('Run Time(s)')
    # for index, file in enumerate(models):
    #     if index < 5:
    #         ax1.bar(ind + (0.15 * index - 37.5), run_times[index, :], color=colors[index], width=0.15)

    ax2 = fig.add_subplot(1, 1, 1)
    if xtick_labels is None:
        xtick_labels = capacities
    ind = np.arange(0, len(xtick_labels)) + 1
    xtickslocs = ind
    ax2.set_xticks(ind)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_title(plot_title)
    ax2.set_xlabel('Capacities')
    ax2.set_ylabel('Test Regret')

    # if is_run_time:
    #     ax2.set_ylabel('Run Time Until Early Stop(s)')

    relu_ppo_regrets = [np.zeros((len(capacities), 5)) for l in layer_params_list[1:]]
    regression_regrets = np.zeros((len(capacities), 5))
    relu_spo_regrets = [np.zeros((len(capacities), 5)) for l in layer_params_list[1:]]
    spo_regrets = np.zeros((len(capacities), 5))
    intopt_regrets = np.zeros((len(capacities), 5))
    for ci, c in enumerate(capacities):
        this_relu_folds = [None for ki in enumerate(kfolds)]
        this_dnl_folds = [None for ki in enumerate(kfolds)]
        this_regression = [None for ki in enumerate(kfolds)]
        this_relu_spo_folds = [None for ki in enumerate(kfolds)]
        this_intopt = [None for ki in enumerate(kfolds)]
        for li, l in enumerate(layer_params_list):
            for pi in param_range:
                number_of_folds = 0

                for ki, k in enumerate(kfolds):
                    this_df = df[ci][li][pi][ki]

                    if this_df is not None:
                        number_of_folds += 1

                        this_fold = np.array(this_df.data).astype(float)
                        if this_regression[ki] is None:
                            this_regression[ki] = this_fold[this_fold[:, 1] == 0]
                        else:
                            this_regression[ki] = np.vstack(
                                (this_regression[ki], this_fold[this_fold[:, 1] == 0]))

                        if li == 0:
                            if this_dnl_folds[ki] is None:
                                this_dnl_folds[ki] = this_fold
                            else:
                                this_dnl_folds[ki] = np.vstack((this_dnl_folds[ki], this_fold))
                        else:
                            if this_relu_folds[ki] is None:
                                this_relu_folds[ki] = this_fold
                            else:
                                this_relu_folds[ki] = np.vstack((this_relu_folds[ki], this_fold))
            if li == 1 or li == 2:
                for ki, k in enumerate(kfolds):
                    this_relu_spo_fold = np.array(df_relu_spo[ci][li - 1][ki].data).astype(float)
                    if this_relu_spo_folds[ki] is None:
                        this_relu_spo_folds[ki] = this_relu_spo_fold
                    else:
                        this_relu_spo_folds[ki] = np.vstack((this_relu_spo_folds[ki], this_relu_spo_fold))

        # Process Regression Folds
        for ki, this_regression_fold in enumerate(this_regression):
            regression_min_val_ind = np.argmin(this_regression_fold[:, RELU_VAL_REGRET_IND])
            regression_min_regret = np.array(
                this_regression_fold[regression_min_val_ind, RELU_REGRET_RATIO_IND])
            regression_regrets[ci, ki] = regression_min_regret

        # Process Relu Folds
        for ki, this_relu_fold in enumerate(this_relu_folds):
            min_val_ind = np.argmin(this_relu_fold[:, RELU_VAL_REGRET_IND])
            min_val = this_relu_fold[min_val_ind, RELU_VAL_REGRET_IND]
            min_regret = np.array(this_relu_fold[min_val_ind, RELU_REGRET_RATIO_IND])
            run_time = np.array(this_relu_fold[min_val_ind, RELU_RUN_TIME_IND])

            relu_ppo_regrets[1][ci, ki] = min_regret

        for ki, this_dnl_fold in enumerate(this_dnl_folds):
            min_val_ind = np.argmin(this_dnl_fold[:, RELU_VAL_REGRET_IND])
            min_val = this_dnl_fold[min_val_ind, RELU_VAL_REGRET_IND]
            min_regret = np.array(this_dnl_fold[min_val_ind, RELU_REGRET_RATIO_IND])
            run_time = np.array(this_relu_fold[min_val_ind, RELU_RUN_TIME_IND])

            relu_ppo_regrets[0][ci, ki] = min_regret
        # Process SPO

        if df_relu_spo is not None:
            for ki, this_spo_fold in enumerate(this_relu_spo_folds):
                this_spo_fold = np.array(this_spo_fold.data).astype(float)
                min_val_ind = np.argmin(this_spo_fold[:, SPO_VAL_REGRET_IND])
                min_val = this_spo_fold[min_val_ind, SPO_VAL_REGRET_IND]
                min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND]) * 100 / this_relu_fold[
                    0, RELU_OBJECTIVE_IND]
                # min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND])
                run_time = np.array(this_spo_fold[min_val_ind, SPO_RUN_TIME_IND])

                relu_spo_regrets[0][ci, ki] = min_regret

        if df_spo is not None:
            for ki, this_spo_fold in enumerate(df_spo[ci]):
                this_spo_fold = np.array(this_spo_fold.data).astype(float)
                min_val_ind = np.argmin(this_spo_fold[:, SPO_VAL_REGRET_IND])
                min_val = this_spo_fold[min_val_ind, SPO_VAL_REGRET_IND]
                min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND]) * 100 / \
                             this_relu_fold[
                                 0, RELU_OBJECTIVE_IND]
                # min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND])
                run_time = np.array(this_spo_fold[min_val_ind, SPO_RUN_TIME_IND])

                spo_regrets[ci, ki] = min_regret

            #Process IntOpt

            for ki, this_intopt_fold in enumerate(df_intopt[ci]):
                this_intopt_fold = np.array(this_intopt_fold.data).astype(float)
                min_val_ind = np.argmin(this_intopt_fold[:, SPO_VAL_REGRET_IND])
                min_val = this_intopt_fold[min_val_ind, SPO_VAL_REGRET_IND]
                min_regret = np.array(this_intopt_fold[min_val_ind, SPO_REGRET_IND]) * 100 / \
                             this_relu_fold[
                                 0, RELU_OBJECTIVE_IND]
                # min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND])
                run_time = np.array(this_intopt_fold[min_val_ind, SPO_RUN_TIME_IND])

                intopt_regrets[ci, ki] = min_regret
            print('hello')
            # if ki == 0:
            #     min_regrets = min_regret
            #     run_times = run_time
            #     min_vals = min_val
            #     min_val_inds = min_val_ind
            # else:
            #     min_regrets = np.vstack((min_regrets, min_regret))
            #     run_times = np.vstack((run_times, run_time))
            #     min_vals = np.vstack((min_vals, min_val))
            #     min_val_inds = np.vstack((min_val_inds, min_val_ind))
            # print(min_regret)

    regrets = np.array([np.mean(relu_ppo_regrets[0], axis=1), np.mean(relu_ppo_regrets[1], axis=1),
                        np.mean(regression_regrets, axis=1),
                        np.mean(spo_regrets, axis=1), np.mean(relu_spo_regrets[0], axis=1), np.mean(intopt_regrets, axis=1)])
    # errors = np.array([np.std(relu_ppo_regrets[0], axis=1), np.std(relu_ppo_regrets[1], axis=1),
    #                    np.std(regression_regrets, axis=1),
    #                    np.std(spo_regrets, axis=1)])

    max_regret = np.max(np.mean(spo_regrets, axis=1))
    for index, regret in enumerate(regrets):
        ind_plot = index
        ax2.bar(ind + ((+0.10 * ind_plot) - 0.22), regrets[index, :], color=facecolors[index],
                width=0.1, hatch=patterns[index]
               )  # ecolor=ecolors[index]

        # if ylim is None:
        #     if is_normalize:
        #         ylim = 2
        #     else:
        #         ylim = max_regret * 1.5
        ax2.set_ylim(0, 6)

    # ax2.xaxis.set_major_locator(mticker.FixedLocator(xtickslocs))
    labels = ['DnL', 'Relu_PPO', 'Ridge Regression', 'SPO-Linear', 'SPO Non Linear', 'IntOpt']
    ax2.legend(labels)
    dest_file_name = "all_knapsack.pdf"
    if is_save:
        plt.savefig('../figs/' + str(dest_file_name))
    plt.show()

def bar_plots_compare_nonlinearity_models(df, capacities, layer_params_list,
                         param_range, kfolds, df_spo, df_relu_spo, xtick_labels=None, is_normalize=False, is_save=True, ylim=None):
    plot_title = "Weighted Knapsack"

    fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ind = np.arange(0, 6)
    # ax1.set_xticks(ind)
    # ax1.set_xticklabels(capacities)
    # ax1.set_title('Run Time vs Sample Points')
    # ax1.set_xlabel('Sample Points')
    # ax1.set_ylabel('Run Time(s)')
    # for index, file in enumerate(models):
    #     if index < 5:
    #         ax1.bar(ind + (0.15 * index - 37.5), run_times[index, :], color=colors[index], width=0.15)

    ax2 = fig.add_subplot(1, 1, 1)
    if xtick_labels is None:
        xtick_labels = capacities
    ind = np.arange(0, len(xtick_labels)) + 1
    xtickslocs = ind
    ax2.set_xticks(ind)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_title(plot_title)
    ax2.set_xlabel('Capacities')
    ax2.set_ylabel('Test Regret')

    # if is_run_time:
    #     ax2.set_ylabel('Run Time Until Early Stop(s)')

    relu_ppo_regrets = [np.zeros((len(capacities), 5)) for l in layer_params_list]
    regression_regrets = np.zeros((len(capacities), 5))
    relu_spo_regrets = [np.zeros((len(capacities), 5)) for l in layer_params_list[1:]]
    spo_regrets = np.zeros((len(capacities), 5))

    for ci, c in enumerate(capacities):
        for li, l in enumerate(layer_params_list):
            this_relu_folds = [None for ki in enumerate(kfolds)]
            this_dnl_folds = [None for ki in enumerate(kfolds)]
            this_regression = [None for ki in enumerate(kfolds)]

            for pi in param_range:
                number_of_folds = 0

                for ki, k in enumerate(kfolds):
                    this_df = df[ci][li][pi][ki]

                    if this_df is not None:
                        number_of_folds += 1

                        this_fold = np.array(this_df.data).astype(float)
                        if this_regression[ki] is None:
                            this_regression[ki] = this_fold[this_fold[:, 1] == 0]
                        else:
                            this_regression[ki] = np.vstack(
                                (this_regression[ki], this_fold[this_fold[:, 1] == 0]))

                        if this_relu_folds[ki] is None:
                            this_relu_folds[ki] = this_fold
                        else:
                            this_relu_folds[ki] = np.vstack((this_relu_folds[ki], this_fold))
            # Process Regression Folds
            for ki, this_regression_fold in enumerate(this_regression):
                regression_min_val_ind = np.argmin(this_regression_fold[:, RELU_VAL_REGRET_IND])
                regression_min_regret = np.array(
                    this_regression_fold[regression_min_val_ind, RELU_REGRET_RATIO_IND])
                regression_regrets[ci, ki] = regression_min_regret

            # Process Relu Folds
            for ki, this_relu_fold in enumerate(this_relu_folds):
                min_val_ind = np.argmin(this_relu_fold[:, RELU_VAL_REGRET_IND])
                min_val = this_relu_fold[min_val_ind, RELU_VAL_REGRET_IND]
                min_regret = np.array(this_relu_fold[min_val_ind, RELU_REGRET_RATIO_IND])
                run_time = np.array(this_relu_fold[min_val_ind, RELU_RUN_TIME_IND])

                relu_ppo_regrets[li][ci, ki] = min_regret

            # Process SPO
            if li == 1 or li == 2:
                if df_relu_spo is not None:
                    for ki, this_spo_fold in enumerate(df_relu_spo[ci][li - 1]):
                        this_spo_fold = np.array(this_spo_fold.data).astype(float)
                        min_val_ind = np.argmin(this_spo_fold[:, SPO_VAL_REGRET_IND])
                        min_val = this_spo_fold[min_val_ind, SPO_VAL_REGRET_IND]
                        min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND]) * 100 / this_relu_fold[
                            0, RELU_OBJECTIVE_IND]
                        # min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND])
                        run_time = np.array(this_spo_fold[min_val_ind, SPO_RUN_TIME_IND])

                        relu_spo_regrets[li][ci, ki] = min_regret

        if df_spo is not None:
            for ki, this_spo_fold in enumerate(df_spo[ci]):
                this_spo_fold = np.array(this_spo_fold.data).astype(float)
                min_val_ind = np.argmin(this_spo_fold[:, SPO_VAL_REGRET_IND])
                min_val = this_spo_fold[min_val_ind, SPO_VAL_REGRET_IND]
                min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND]) * 100 / \
                             this_relu_fold[
                                 0, RELU_OBJECTIVE_IND]
                # min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND])
                run_time = np.array(this_spo_fold[min_val_ind, SPO_RUN_TIME_IND])

                spo_regrets[ci, ki] = min_regret
            print('hello')
            # if ki == 0:
            #     min_regrets = min_regret
            #     run_times = run_time
            #     min_vals = min_val
            #     min_val_inds = min_val_ind
            # else:
            #     min_regrets = np.vstack((min_regrets, min_regret))
            #     run_times = np.vstack((run_times, run_time))
            #     min_vals = np.vstack((min_vals, min_val))
            #     min_val_inds = np.vstack((min_val_inds, min_val_ind))
            # print(min_regret)

    regrets = np.array([np.mean(relu_ppo_regrets[0], axis=1), np.mean(relu_ppo_regrets[1], axis=1), np.mean(relu_ppo_regrets[2], axis=1), np.mean(relu_ppo_regrets[3], axis=1),
                        np.mean(regression_regrets, axis=1),
                        np.mean(spo_regrets, axis=1), np.mean(relu_spo_regrets[1], axis=1),np.mean(relu_spo_regrets[2], axis=1)])
    # errors = np.array([np.std(relu_ppo_regrets[0], axis=1), np.std(relu_ppo_regrets[1], axis=1), np.std(relu_ppo_regrets[2], axis=1), np.std(relu_ppo_regrets[3], axis=1),
    #                    np.std(regression_regrets, axis=1),
    #                    np.std(spo_regrets, axis=1)])

    max_regret = np.max(np.mean(spo_regrets, axis=1))
    for index, regret in enumerate(regrets):
        ind_plot = index
        # ax2.bar(ind + ((+0.10 * ind_plot) - 0.22), regrets[index, :], yerr=errors[index, :], color=facecolors[index],
        #         width=0.1, hatch=patterns[index],
        #         error_kw=dict(lw=0.5, capsize=1.5, capthick=1))  # ecolor=ecolors[index]

        ax2.bar(ind + ((+0.10 * ind_plot) - 0.22), regrets[index, :], color=facecolors[index],
                width=0.1, hatch=patterns[index],
                error_kw=dict(lw=0.5, capsize=1.5, capthick=1))  # ecolor=ecolors[index]

        # if ylim is None:
        #     if is_normalize:
        #         ylim = 2
        #     else:
        #         ylim = max_regret * 1.5
        ax2.set_ylim(0, 6)

    # ax2.xaxis.set_major_locator(mticker.FixedLocator(xtickslocs))
    labels = ['DnL', 'Relu_PPO(1)', "Relu_PPO(2)", "Relu_PPO(3)", 'Regression', 'SPO-Relax', 'ReLu SPO1', 'ReLu SPO2']
    ax2.legend(labels)

    dest_file_name = "non_linearity.pdf"
    if is_save:
        plt.savefig('../figs/' + str(dest_file_name))
    plt.show()

def bar_plots_compare_nonlinearity_models_spo3(df, capacities, layer_params_list,
                         param_range, kfolds, df_spo, df_relu_spo, xtick_labels=None, is_normalize=False, is_save=True, ylim=None):
    plot_title = "Weighted Knapsack"

    fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ind = np.arange(0, 6)
    # ax1.set_xticks(ind)
    # ax1.set_xticklabels(capacities)
    # ax1.set_title('Run Time vs Sample Points')
    # ax1.set_xlabel('Sample Points')
    # ax1.set_ylabel('Run Time(s)')
    # for index, file in enumerate(models):
    #     if index < 5:
    #         ax1.bar(ind + (0.15 * index - 37.5), run_times[index, :], color=colors[index], width=0.15)

    ax2 = fig.add_subplot(1, 1, 1)
    if xtick_labels is None:
        xtick_labels = capacities
    ind = np.arange(0, len(xtick_labels)) + 1
    xtickslocs = ind
    ax2.set_xticks(ind)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_title(plot_title)
    ax2.set_xlabel('Capacities')
    ax2.set_ylabel('Test Regret')

    # if is_run_time:
    #     ax2.set_ylabel('Run Time Until Early Stop(s)')

    relu_ppo_regrets = [np.zeros((len(capacities), 5)) for l in layer_params_list]
    regression_regrets = np.zeros((len(capacities), 5))
    relu_spo_regrets = [np.zeros((len(capacities), 5)) for l in layer_params_list[1:]]
    spo_regrets = np.zeros((len(capacities), 5))

    for ci, c in enumerate(capacities):
        for li, l in enumerate(layer_params_list):
            this_relu_folds = [None for ki in enumerate(kfolds)]
            this_dnl_folds = [None for ki in enumerate(kfolds)]
            this_regression = [None for ki in enumerate(kfolds)]

            for pi in param_range:
                number_of_folds = 0

                for ki, k in enumerate(kfolds):
                    this_df = df[ci][li][pi][ki]

                    if this_df is not None:
                        number_of_folds += 1

                        this_fold = np.array(this_df.data).astype(float)
                        if this_regression[ki] is None:
                            this_regression[ki] = this_fold[this_fold[:, 1] == 0]
                        else:
                            this_regression[ki] = np.vstack(
                                (this_regression[ki], this_fold[this_fold[:, 1] == 0]))

                        if this_relu_folds[ki] is None:
                            this_relu_folds[ki] = this_fold
                        else:
                            this_relu_folds[ki] = np.vstack((this_relu_folds[ki], this_fold))
            # Process Regression Folds
            for ki, this_regression_fold in enumerate(this_regression):
                regression_min_val_ind = np.argmin(this_regression_fold[:, RELU_VAL_REGRET_IND])
                regression_min_regret = np.array(
                    this_regression_fold[regression_min_val_ind, RELU_REGRET_RATIO_IND])
                regression_regrets[ci, ki] = regression_min_regret

            # Process Relu Folds
            for ki, this_relu_fold in enumerate(this_relu_folds):
                min_val_ind = np.argmin(this_relu_fold[:, RELU_VAL_REGRET_IND])
                min_val = this_relu_fold[min_val_ind, RELU_VAL_REGRET_IND]
                min_regret = np.array(this_relu_fold[min_val_ind, RELU_REGRET_RATIO_IND])
                run_time = np.array(this_relu_fold[min_val_ind, RELU_RUN_TIME_IND])

                relu_ppo_regrets[li][ci, ki] = min_regret

            # Process SPO
            if li != 0:
                if df_relu_spo is not None:
                    for ki, this_spo_fold in enumerate(df_relu_spo[ci][li - 1]):
                        this_spo_fold = np.array(this_spo_fold.data).astype(float)
                        min_val_ind = np.argmin(this_spo_fold[:, SPO_VAL_REGRET_IND])
                        min_val = this_spo_fold[min_val_ind, SPO_VAL_REGRET_IND]
                        min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND]) * 100 / this_relu_fold[
                            0, RELU_OBJECTIVE_IND]
                        # min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND])
                        run_time = np.array(this_spo_fold[min_val_ind, SPO_RUN_TIME_IND])

                        relu_spo_regrets[li-1][ci, ki] = min_regret

        if df_spo is not None:
            for ki, this_spo_fold in enumerate(df_spo[ci]):
                this_spo_fold = np.array(this_spo_fold.data).astype(float)
                min_val_ind = np.argmin(this_spo_fold[:, SPO_VAL_REGRET_IND])
                min_val = this_spo_fold[min_val_ind, SPO_VAL_REGRET_IND]
                min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND]) * 100 / \
                             this_relu_fold[
                                 0, RELU_OBJECTIVE_IND]
                # min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND])
                run_time = np.array(this_spo_fold[min_val_ind, SPO_RUN_TIME_IND])

                spo_regrets[ci, ki] = min_regret
            print('hello')
            # if ki == 0:
            #     min_regrets = min_regret
            #     run_times = run_time
            #     min_vals = min_val
            #     min_val_inds = min_val_ind
            # else:
            #     min_regrets = np.vstack((min_regrets, min_regret))
            #     run_times = np.vstack((run_times, run_time))
            #     min_vals = np.vstack((min_vals, min_val))
            #     min_val_inds = np.vstack((min_val_inds, min_val_ind))
            # print(min_regret)

    regrets = np.array([np.mean(relu_ppo_regrets[0], axis=1), np.mean(relu_ppo_regrets[1], axis=1), np.mean(relu_ppo_regrets[2], axis=1), np.mean(relu_ppo_regrets[3], axis=1),
                        np.mean(regression_regrets, axis=1),
                        np.mean(spo_regrets, axis=1), np.mean(relu_spo_regrets[0], axis=1),np.mean(relu_spo_regrets[1], axis=1), np.mean(relu_spo_regrets[2], axis=1)])
    # errors = np.array([np.std(relu_ppo_regrets[0], axis=1), np.std(relu_ppo_regrets[1], axis=1), np.std(relu_ppo_regrets[2], axis=1), np.std(relu_ppo_regrets[3], axis=1),
    #                    np.std(regression_regrets, axis=1),
    #                    np.std(spo_regrets, axis=1)])

    max_regret = np.max(np.mean(spo_regrets, axis=1))
    for index, regret in enumerate(regrets):
        ind_plot = index
        # ax2.bar(ind + ((+0.10 * ind_plot) - 0.22), regrets[index, :], yerr=errors[index, :], color=facecolors[index],
        #         width=0.1, hatch=patterns[index],
        #         error_kw=dict(lw=0.5, capsize=1.5, capthick=1))  # ecolor=ecolors[index]

        ax2.bar(ind + ((+0.10 * ind_plot) - 0.22), regrets[index, :], color=facecolors[index],
                width=0.1, hatch=patterns[index],
                error_kw=dict(lw=0.5, capsize=1.5, capthick=1))  # ecolor=ecolors[index]

        # if ylim is None:
        #     if is_normalize:
        #         ylim = 2
        #     else:
        #         ylim = max_regret * 1.5
        ax2.set_ylim(0, 6)

    # ax2.xaxis.set_major_locator(mticker.FixedLocator(xtickslocs))
    labels = ['DnL', 'Relu_PPO(1)', "Relu_PPO(2)", "Relu_PPO(3)", 'Regression', 'SPO-Relax', 'ReLu SPO1', 'ReLu SPO2', 'ReLu SPO2']
    ax2.legend(labels)

    dest_file_name = "non_linearity_spom3.pdf"
    if is_save:
        plt.savefig('../figs/' + str(dest_file_name))
    plt.show()



def plot_knapsack_plots(is_spo=True):
    kfolds = [0, 1, 2, 3, 4]
    capacities = [12, 24, 48, 96, 196, 220]
    # capacities = [12, 24, 48, 72]
    # capacities = [12, 24, 48, 72, 96, 120, 144, 172, 196, 220]
    jobs = [i + 1 for i in range(10)]
    machines = [i + 1 for i in range(3)]
    param_range = [0, 1, 2]
    models_list = ['ReLuDNL']
    layer_params_list = ['01', "0501", "0901", "050501"]
    layer_params_list_spo = ['01', "0501", "0901", "050501"]

    file_name_list = [
        [[generate_file_name_ReLu_knapsack(capacity=capacity, fold=kfold,
                                           layer_params=layer_params) for kfold in kfolds] for
         layer_params in layer_params_list] for capacity in capacities]

    df = combine_data_scheduling(file_name_list, capacities, layer_params_list, kfolds)

    if is_spo:
        file_name_list_spo = [[generate_file_name_SPO_knapsack(capacity=capacity, fold=kfold)
                               for kfold in kfolds] for capacity in capacities]

        df_spo = get_spo_df(file_name_list_spo, capacities, kfolds)

        file_name_list_relu_spo = [[[generate_file_name_non_linear_SPO_knapsack(capacity=capacity,
                                                                                fold=kfold,
                                                                                layer_params=layer_params) for kfold in
                                     kfolds] for
                                    layer_params in layer_params_list[1:]] for capacity in capacities]

    file_name_list_intopt = [[generate_file_name_intopt_knapsack(capacity=capacity, fold=kfold)
                               for kfold in kfolds] for capacity in capacities]
    df_intopt = get_intopt_df(file_name_list_intopt, capacities, kfolds)

    df_nl_spo = get_non_linear_spo_df(file_name_list_relu_spo, capacities, layer_params_list[1:-1], kfolds)

    # df_nl_spo = None

    bar_plots_all_models(capacities=capacities, layer_params_list=layer_params_list, param_range=param_range,
                         kfolds=kfolds, df=df, df_spo=df_spo, df_relu_spo=df_nl_spo, df_intopt=df_intopt)
    bar_plots_compare_nonlinearity_models(capacities=capacities, layer_params_list=layer_params_list, param_range=param_range,
                         kfolds=kfolds, df=df, df_spo=df_spo, df_relu_spo=df_nl_spo)


    capacities = [12, 24, 48, 72]
    file_name_list_relu_spo = [[[generate_file_name_non_linear_SPO_knapsack(capacity=capacity,
                                                                            fold=kfold,
                                                                            layer_params=layer_params) for kfold in
                                 kfolds] for
                                layer_params in layer_params_list[1:]] for capacity in capacities]

    df_nl_spo = get_non_linear_spo_df(file_name_list_relu_spo, capacities, layer_params_list[1:], kfolds)

    bar_plots_compare_nonlinearity_models_spo3(capacities=capacities, layer_params_list=layer_params_list,
                                          param_range=param_range,
                                          kfolds=kfolds, df=df, df_spo=df_spo, df_relu_spo=df_nl_spo)

if __name__ == "__main__":
    plot_knapsack_plots()
    # parse_file_ReLu()
