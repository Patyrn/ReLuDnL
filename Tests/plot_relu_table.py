import copy
from itertools import zip_longest

import numpy as np
from tabulate import tabulate
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

SPO_VAL_REGRET_IND = 1
SPO_RUN_TIME_IND = 5
SPO_REGRET_IND = 3

INTOPT_VAL_REGRET_IND = 0
INTOPT_RUN_TIME_IND = 2
INTOPT_REGRET_IND = 1


def parse_file_ReLu(f_name="Iconsched_l30k0_ReLuDNL_l01.csv", f_folder=None):
    if f_folder is None:
        f_folder = "Tests/{}/{}".format(ICON_SCHEDULING_FOLDER, "ReLuDNL")
    df = read_file(filename=f_name, folder_path=f_folder)
    df = seperate_models(df)
    return df


def parse_file_spo(f_name="Load30SPOmax_spartan_kfold1.csv", f_folder=None):
    if f_folder is None:
        f_folder = "Tests/{}/{}".format(ICON_SCHEDULING_FOLDER, "SPORelax")
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


def combine_data_scheduling(file_name_list, machines, jobs, layer_params_list, kfolds):
    df_temp = [[[[[None for k in range(len(kfolds))] for p in range(3)] for
                 l in range(len(layer_params_list))] for j in range(len(jobs))] for m in range(len(machines))]

    for mi, m in enumerate(machines):
        for ji, j in enumerate(jobs):
            for li, l in enumerate(layer_params_list):
                param_range = set()
                tmp_data = [[] for ki in enumerate(kfolds)]
                for ki, k in enumerate(kfolds):
                    data = parse_file_ReLu(f_name=file_name_list[mi][ji][li][ki])
                    for test_data in data:
                        test_data.opt = "m{}j{}".format(m, j)
                        test_data.l = l
                        test_data.k = k
                        param_range.add(int(test_data.p))
                    tmp_data[ki].extend(data)
                param_range = sorted(param_range)
                # print('layer')
                for ki, k in enumerate(kfolds):
                    for data in tmp_data[ki]:
                        try:
                            df_temp[mi][ji][li][param_range.index(int(data.p))][ki] = data
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


def get_spo_df(file_name_list, machines, jobs, kfolds):
    df_temp = [[[None for k in range(len(kfolds))] for j in range(len(jobs))] for m in range(len(machines))]

    for mi, m in enumerate(machines):
        for ji, j in enumerate(jobs):
            for ki, k in enumerate(kfolds):
                data = parse_file_spo(f_name=file_name_list[mi][ji][ki])
                data = [arr[0:6] for arr in data[1:]]
                df_temp[mi][ji][ki] = test_data(data=data[1:], opt="m{}j{}".format(m, j), k=k, )
    return df_temp

def get_intopt_df(file_name_list, machines, jobs, kfolds):
    df_temp = [[[None for k in range(len(kfolds))] for j in range(len(jobs))] for m in range(len(machines))]

    for mi, m in enumerate(machines):
        for ji, j in enumerate(jobs):
            for ki, k in enumerate(kfolds):
                data = parse_file_intopt(f_name=file_name_list[mi][ji][ki])
                data = [list(map(arr.__getitem__, [0, 2, 7])) for arr in data[1:]]
                df_temp[mi][ji][ki] = test_data(data=data[1:], opt="m{}j{}".format(m, j), k=k)
    return df_temp

def get_non_linear_spo_df(file_name_list, machines, jobs, layer_params_list, kfolds):
    df_temp = [[[[None for k in range(len(kfolds))] for
                 l in range(len(layer_params_list))] for j in range(len(jobs))] for m in range(len(machines))]

    for mi, m in enumerate(machines):
        for ji, j in enumerate(jobs):
            for li, l in enumerate(layer_params_list):
                for ki, k in enumerate(kfolds):
                    data = parse_file_spo(f_name=file_name_list[mi][ji][li][ki])
                    data = [arr[0:6] for arr in data[1:]]
                    df_temp[mi][ji][li][ki] = test_data(data=data[1:], opt="m{}j{}".format(m, j), k=k, )
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
    f_name = "{}_l{}c{}_{}_l{}.csv".format(prefix, capacity, fold, model, layer_params)
    return f_name


def generate_file_name_SPO_scheduling(prefix='Load', load=30, fold=0):
    f_name = "{}{}SPOmax_spartan_kfold{}.csv".format(prefix, load, fold)
    return f_name


def generate_file_name_non_linear_SPO_scheduling(load=30, fold=0, layer_params='0901'):
    f_name = "Iconsched_l{}k{}_SPO_l09{}.csv".format(load, fold, layer_params)
    return f_name


def generate_file_name_SPO_knapsack(capacity=12, fold=0):
    f_name = "gurobi_knapsack_SPOk{}_c{}.csv".format(capacity, fold)
    return f_name


def generate_file_name_non_linear_SPO_knapsack(capacity=12, fold=0, layer_params='0901'):
    f_name = "Iconknap_c{}k{}_SPO_l09{}.csv".format(capacity, fold, layer_params)
    return f_name

def generate_file_name_intopt_scheduling(load=30, fold=0):
    f_name = "0lintoptl{}k{}.csv".format(load, fold)
    return f_name

def plot_knapsack_table():
    print('hi')


def print_divider_table(df, machines, jobs, layer_params_list, param_range, kfolds):
    first_row = ["Jobs", "M3 Relu-DNL  D10", "M3 Relu-DNL  R D5 5", "M3 Relu-DNL  R D1",
                 "M4 Relu-DNL  R D10", "M4 Relu-DNL  R D5 5", "M4 Relu-DNL  R D1"]
    myTable = PrettyTable(first_row)
    tabulate_df = [[None for j in jobs] for m in machines]
    for mi, m in enumerate(machines):
        machine_row = ["******M{}******".format(m) for i in range(len(first_row))]
        myTable.add_row(machine_row)
        for ji, j in enumerate(jobs):
            row_str = [str(ji + 1)]
            tabulate_df[mi][ji] = [str(ji + 1)]
            for li, l in enumerate(layer_params_list):
                if li != 0 and li != 1:
                    this_regrets = []
                    this_run_times = []
                    this_regression = [None for ki in enumerate(kfolds)]
                    for pi in param_range:
                        number_of_folds = 0

                        for ki, k in enumerate(kfolds):
                            this_df = df[mi][ji][li][pi][ki]

                            if this_df is not None:
                                number_of_folds += 1

                                this_fold = np.array(this_df.data).astype(float)
                                if this_regression[ki] is None:
                                    this_regression[ki] = this_fold[this_fold[:, 1] == 0]
                                else:
                                    this_regression[ki] = np.vstack(
                                        (this_regression[ki], this_fold[this_fold[:, 1] == 0]))

                                min_val_ind = np.argmin(this_fold[:, RELU_VAL_REGRET_IND])
                                min_regret = np.array(this_fold[min_val_ind, RELU_REGRET_RATIO_IND])
                                run_time = np.array(this_fold[min_val_ind, RELU_RUN_TIME_IND])

                                if number_of_folds == 1:
                                    min_regrets = min_regret
                                    run_times = run_time
                                else:
                                    min_regrets = np.vstack((min_regrets, min_regret))
                                    run_times = np.vstack((run_times, run_time))

                                    # print(min_regret)
                        this_regrets.append(np.mean(min_regrets))
                        this_run_times.append(np.mean(run_times))
                        this_str = ["{}\% ({:.1e})".format(str(abs(round(np.mean(min_regrets), 2))),
                                                           (np.mean(run_times).astype(int)))]
                        if number_of_folds != 5:
                            this_str[0] += "!"
                        tabulate_df[mi][ji].extend(this_str)
                        row_str.extend(this_str)

                    # # Process Regression
                    #
                    # for ki, this_regression_fold in enumerate(this_regression):
                    #
                    #     regression_min_val_ind = np.argmin(this_regression_fold[:, RELU_VAL_REGRET_IND])
                    #     regression_min_regret = np.array(
                    #         this_regression_fold[regression_min_val_ind, RELU_REGRET_RATIO_IND])
                    #     if ki == 0:
                    #         regression_regrets = regression_min_regret
                    #     else:
                    #         regression_regrets = np.vstack((regression_regrets, regression_min_regret))

                    # this_regrets.insert(0, np.mean(regression_regrets))
                    # this_run_times.insert(0, float("inf"))
                    # row_str.insert(li * 4 + 1, "{}%".format(str(abs(round(np.mean(regression_regrets), 2)))))
                    # tabulate_df[mi][ji].extend(this_str)
                    max_regret_ind_str = np.argmax(np.array(this_regrets))
                    min_runtime_ind_str = np.argmin(np.array(this_run_times))

                    tabulate_df[mi][ji][((li - 2) * 3) + max_regret_ind_str + 1] = "\\textbf{{{}}}".format(
                        tabulate_df[mi][ji][((li - 2) * 3) + max_regret_ind_str + 1])
                    row_str[((li - 2) * 3) + max_regret_ind_str + 1] = row_str[((
                                                                                            li - 2) * 3) + max_regret_ind_str + 1] + "*"
                    row_str[((li - 2) * 3) + min_runtime_ind_str + 1] = row_str[((
                                                                                             li - 2) * 3) + min_runtime_ind_str + 1] + "^"

            myTable.add_row(row_str)
    print(myTable)
    for mi, m in enumerate(machines):
        print(tabulate(np.array(tabulate_df[mi]), tablefmt="latex_raw", floatfmt=".2f"))


def print_nonlinearity_table(df, machines, jobs, layer_params_list, param_range, kfolds):
    first_row = ["Jobs", "M1 Reg", "M1 R D10", "M1 R D5 5", "M1 R D1",
                 "M2 Reg", "M2 R D10", "M2 R D5 5", "M2 R D1",
                 "M3 Reg", "M3 R D10", "M3 R D5 5", "M3 R D1",
                 "M4 Reg", "M4 R D10", "M4 R D5 5", "M4 R D1"]
    tabulate_df = [[None for j in jobs] for m in machines]
    myTable = PrettyTable(first_row)
    for mi, m in enumerate(machines):
        machine_row = ["******M{}******".format(m) for i in range(len(first_row))]
        myTable.add_row(machine_row)
        for ji, j in enumerate(jobs):
            row_str = [str(ji + 1)]
            tabulate_df[mi][ji] = [str(ji + 1)]
            for li, l in enumerate(layer_params_list):
                this_regrets = []
                this_run_times = []
                this_regression = [None for ki in enumerate(kfolds)]
                for pi in param_range:
                    number_of_folds = 0

                    for ki, k in enumerate(kfolds):
                        this_df = df[mi][ji][li][pi][ki]

                        if this_df is not None:
                            number_of_folds += 1

                            this_fold = np.array(this_df.data).astype(float)
                            if this_regression[ki] is None:
                                this_regression[ki] = this_fold[this_fold[:, 1] == 0]
                            else:
                                this_regression[ki] = np.vstack((this_regression[ki], this_fold[this_fold[:, 1] == 0]))

                            min_val_ind = np.argmin(this_fold[:, RELU_VAL_REGRET_IND])
                            min_regret = np.array(this_fold[min_val_ind, RELU_REGRET_RATIO_IND])
                            run_time = np.array(this_fold[min_val_ind, RELU_RUN_TIME_IND])

                            if number_of_folds == 1:
                                min_regrets = min_regret
                                run_times = run_time
                            else:
                                min_regrets = np.vstack((min_regrets, min_regret))
                                run_times = np.vstack((run_times, run_time))

                            # print(min_regret)
                    this_regrets.append(np.mean(min_regrets))
                    this_run_times.append(np.mean(run_times))
                    tabulate_df[mi][ji] = ["{}\%".format(str(abs(round(np.mean(min_regrets), 2))))]
                    this_str = ["{}% ({})".format(str(abs(round(np.mean(min_regrets), 2))),
                                                  str(round(np.mean(run_times), 3)))]
                    if number_of_folds != 5:
                        this_str[0] += "!"

                    row_str.extend(this_str)

                this_run_times.insert(0, float("inf"))

                max_regret_ind_str = np.argmax(np.array(this_regrets))
                min_runtime_ind_str = np.argmin(np.array(this_run_times))

                tabulate_df[mi][ji][(li * 4) + max_regret_ind_str + 1] = "\\textbf{{{}}}".format(
                    tabulate_df[mi][ji][(li * 3) + max_regret_ind_str + 1])
                row_str[(li * 3) + max_regret_ind_str + 1] = row_str[(li * 3) + max_regret_ind_str + 1] + "*"
                row_str[(li * 3) + min_runtime_ind_str + 1] = row_str[(li * 3) + min_runtime_ind_str + 1] + "^"

        myTable.add_row(row_str)
    print(myTable)


def print_simple_regrets_table(df, machines, jobs, layer_params_list, param_range, kfolds, df_relu_spo, df_intopt,df_spo=None):
    if df_spo is not None:
        first_row = ["Jobs", "M1 Reg", "DNL", "ReLu DNL", "SPO", "Relu SPO", "IntOpt"]
    else:
        first_row = ["Jobs", "M1 Reg", "Linear", "Non Linear"]
    myTable = PrettyTable(first_row)
    tabulate_df = [[None for j in jobs] for m in machines]
    for mi, m in enumerate(machines):
        machine_row = ["******M{}******".format(m) for i in range(len(first_row))]
        myTable.add_row(machine_row)
        for ji, j in enumerate(jobs):
            row_str = [str(ji + 1)]
            tabulate_df[mi][ji] = [str(ji + 1)]
            this_regrets = []
            this_run_times = []
            this_min_vals = []
            this_min_val_inds = []
            this_relu_folds = [None for ki in enumerate(kfolds)]
            this_dnl_folds = [None for ki in enumerate(kfolds)]
            this_regression = [None for ki in enumerate(kfolds)]
            this_spo_relu_folds = [None for ki in enumerate(kfolds)]
            this_intopt_folds = [None for ki in enumerate(kfolds)]
            for li, l in enumerate(layer_params_list):

                for pi in param_range:
                    number_of_folds = 0

                    for ki, k in enumerate(kfolds):
                        this_df = df[mi][ji][li][pi][ki]

                        if this_df is not None:
                            number_of_folds += 1

                            this_fold = np.array(this_df.data).astype(float)
                            if this_regression[ki] is None:
                                this_regression[ki] = this_fold[this_fold[:, 1] == 0]
                            else:
                                this_regression[ki] = np.vstack((this_regression[ki], this_fold[this_fold[:, 1] == 0]))

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
                        if this_regression[ki] is None:
                            this_regression[ki] = this_fold[this_fold[:, 1] == 0]
                        else:
                            this_regression[ki] = np.vstack((this_regression[ki], this_fold[this_fold[:, 1] == 0]))

                for ki, k in enumerate(kfolds):
                    if li > 0:
                        this_spo_df = df_relu_spo[mi][ji][li - 1][ki]
                        this_fold = np.array(this_spo_df.data).astype(float)
                        if this_spo_relu_folds[ki] is None:
                            this_spo_relu_folds[ki] = this_fold
                        else:
                            this_spo_relu_folds[ki] = np.vstack((this_spo_relu_folds[ki], this_fold))

            # Process Regression Folds
            for ki, this_regression_fold in enumerate(this_regression):

                regression_min_val_ind = np.argmin(this_regression_fold[:, RELU_VAL_REGRET_IND])
                regression_min_regret = np.array(
                    this_regression_fold[regression_min_val_ind, RELU_REGRET_RATIO_IND])
                if ki == 0:
                    regression_regrets = regression_min_regret
                else:
                    regression_regrets = np.vstack((regression_regrets, regression_min_regret))

            this_str = ["{}\%".format(str(abs(round(np.mean(regression_regrets), 2))))]
            row_str.extend(this_str)
            tabulate_df[mi][ji].extend(this_str)

            this_regrets.insert(0, np.mean(regression_regrets))
            this_run_times.insert(0, float("inf"))

            # Process DNL folds
            for ki, this_relu_fold in enumerate(this_dnl_folds):
                min_val_ind = np.argmin(this_relu_fold[:, RELU_VAL_REGRET_IND])
                min_val = this_relu_fold[min_val_ind, RELU_VAL_REGRET_IND]
                min_regret = np.array(this_relu_fold[min_val_ind, RELU_REGRET_RATIO_IND])
                run_time = np.array(this_relu_fold[min_val_ind, RELU_RUN_TIME_IND])

                if ki == 0:
                    min_regrets = min_regret
                    run_times = run_time
                    min_vals = min_val
                    min_val_inds = min_val_ind
                else:
                    min_regrets = np.vstack((min_regrets, min_regret))
                    run_times = np.vstack((run_times, run_time))
                    min_vals = np.vstack((min_vals, min_val))
                    min_val_inds = np.vstack((min_val_inds, min_val_ind))
                    # print(min_regret)
            this_str = ["{}\%".format(str(abs(round(np.mean(min_regrets), 2))))]
            row_str.extend(this_str)
            tabulate_df[mi][ji].extend(this_str)

            this_regrets.append(np.mean(min_regrets))
            this_run_times.append(np.mean(run_times))
            # Process Relu Folds
            for ki, this_relu_fold in enumerate(this_relu_folds):
                min_val_ind = np.argmin(this_relu_fold[:, RELU_VAL_REGRET_IND])
                min_val = this_relu_fold[min_val_ind, RELU_VAL_REGRET_IND]
                min_regret = np.array(this_relu_fold[min_val_ind, RELU_REGRET_RATIO_IND])
                run_time = np.array(this_relu_fold[min_val_ind, RELU_RUN_TIME_IND])

                if ki == 0:
                    min_regrets = min_regret
                    run_times = run_time
                    min_vals = min_val
                    min_val_inds = min_val_ind
                else:
                    min_regrets = np.vstack((min_regrets, min_regret))
                    run_times = np.vstack((run_times, run_time))
                    min_vals = np.vstack((min_vals, min_val))
                    min_val_inds = np.vstack((min_val_inds, min_val_ind))
                    # print(min_regret)
            this_str = ["{}\%".format(str(abs(round(np.mean(min_regrets), 2))))]
            tabulate_df[mi][ji].extend(this_str)

            row_str.extend(this_str)
            this_regrets.append(np.mean(min_regrets))
            this_run_times.append(np.mean(run_times))

            # Process SPO
            if df_spo is not None:
                for ki, this_spo_fold in enumerate(df_spo[mi][ji]):
                    this_spo_fold = np.array(this_spo_fold.data).astype(float)
                    min_val_ind = np.argmin(this_spo_fold[:, SPO_VAL_REGRET_IND])
                    min_val = this_spo_fold[min_val_ind, SPO_VAL_REGRET_IND]
                    min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND]) * 100 / this_relu_fold[
                        0, RELU_OBJECTIVE_IND]
                    # min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND])
                    run_time = np.array(this_spo_fold[min_val_ind, SPO_RUN_TIME_IND])

                    if ki == 0:
                        min_regrets = min_regret
                        run_times = run_time
                        min_vals = min_val
                        min_val_inds = min_val_ind
                    else:
                        min_regrets = np.vstack((min_regrets, min_regret))
                        run_times = np.vstack((run_times, run_time))
                        min_vals = np.vstack((min_vals, min_val))
                        min_val_inds = np.vstack((min_val_inds, min_val_ind))
                        # print(min_regret)

                # this_str = ["{}% ({})".format(str(abs(round(np.mean(min_regrets), 2))),
                #                               str(round(np.mean(run_times), 3)))]

                this_str = ["{}\%".format(str(abs(round(np.mean(min_regrets), 2))))]

                tabulate_df[mi][ji].extend(this_str)
                row_str.extend(this_str)
                this_regrets.append(np.mean(min_regrets))
                this_run_times.append(np.mean(run_times))

            # Process SPO
            if df_relu_spo is not None:
                for ki, this_spo_fold in enumerate(this_spo_relu_folds):
                    this_spo_fold = np.array(this_spo_fold.data).astype(float)
                    min_val_ind = np.argmin(this_spo_fold[:, SPO_VAL_REGRET_IND])
                    min_val = this_spo_fold[min_val_ind, SPO_VAL_REGRET_IND]
                    min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND]) * 100 / \
                                 this_relu_fold[0, RELU_OBJECTIVE_IND]
                    # min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND])
                    run_time = np.array(this_spo_fold[min_val_ind, SPO_RUN_TIME_IND])

                    if ki == 0:
                        min_regrets = min_regret
                        run_times = run_time
                        min_vals = min_val
                        min_val_inds = min_val_ind
                    else:
                        min_regrets = np.vstack((min_regrets, min_regret))
                        run_times = np.vstack((run_times, run_time))
                        min_vals = np.vstack((min_vals, min_val))
                        min_val_inds = np.vstack((min_val_inds, min_val_ind))
                        # print(min_regret)

                this_str = ["{}\%".format(str(abs(round(np.mean(min_regrets), 2))))]
                tabulate_df[mi][ji].extend(this_str)
                row_str.extend(this_str)
                this_regrets.append(np.mean(min_regrets))
                this_run_times.append(np.mean(run_times))

            #Process IntOpt
            for ki, this_intopt_fold in enumerate(df_intopt[mi][ji]):
                    this_intopt_fold = np.array(this_intopt_fold.data).astype(float)
                    min_val_ind = np.argmin(this_intopt_fold[:, INTOPT_VAL_REGRET_IND])
                    min_val = this_intopt_fold[min_val_ind, INTOPT_VAL_REGRET_IND]
                    min_regret = np.array(this_intopt_fold[min_val_ind, INTOPT_REGRET_IND]) * 100 / this_relu_fold[
                        0, RELU_OBJECTIVE_IND]
                    # min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND])
                    run_time = np.array(this_intopt_fold[min_val_ind, INTOPT_RUN_TIME_IND])

                    if ki == 0:
                        min_regrets = min_regret
                        run_times = run_time
                        min_vals = min_val
                        min_val_inds = min_val_ind
                    else:
                        min_regrets = np.vstack((min_regrets, min_regret))
                        run_times = np.vstack((run_times, run_time))
                        min_vals = np.vstack((min_vals, min_val))
                        min_val_inds = np.vstack((min_val_inds, min_val_ind))
                        # print(min_regret)

                    # this_str = ["{}% ({})".format(str(abs(round(np.mean(min_regrets), 2))),
                    #                               str(round(np.mean(run_times), 3)))]

            this_str = ["{}\%".format(str(abs(round(np.mean(min_regrets), 2))))]

            tabulate_df[mi][ji].extend(this_str)
            row_str.extend(this_str)
            this_regrets.append(np.mean(min_regrets))
            this_run_times.append(np.mean(run_times))
            # Process Regression

            max_regret_ind_str = np.argmax(np.array(this_regrets))
            min_runtime_ind_str = np.argmin(np.array(this_run_times))

            row_str[max_regret_ind_str + 1] = row_str[max_regret_ind_str + 1] + "*"
            tabulate_df[mi][ji][max_regret_ind_str + 1] = "\\textbf{{{}}}".format(
                tabulate_df[mi][ji][max_regret_ind_str + 1])
            # row_str[min_runtime_ind_str + 1] = row_str[min_runtime_ind_str + 1] + "^"

            myTable.add_row(row_str)
    print(myTable)
    print(myTable)
    for mi, m in enumerate(machines):
        print(tabulate(np.array(tabulate_df[mi]), tablefmt="latex_raw", floatfmt=".2f"))


def print_simple_nonlinearity_vs_regression_table(df, machines, jobs, layer_params_list, param_range, kfolds,
                                                  df_spo=None, df_relu_spo=None):
    tabulate_df = [[None for j in jobs] for m in machines]
    if df_spo is not None:
        first_row = ["Jobs", "M1 Reg", "M1 ReLu", "M1 SPO", "M2 Reg", "M2 ReLu", "M2 SPO", "M3 Reg", "M3 ReLu",
                     "M3 SPO"]
    else:
        first_row = ["Jobs", "M1 Reg", "M1 ReLu", "M3 Reg", "M3 ReLu"
            , "M4 Reg", "M4 Non Linear"]
    myTable = PrettyTable(first_row)
    for mi, m in enumerate(machines):
        machine_row = ["******M{}******".format(m) for i in range(len(first_row))]
        myTable.add_row(machine_row)
        for ji, j in enumerate(jobs):
            row_str = [str(ji + 1)]
            tabulate_df[mi][ji] = [str(ji + 1)]
            for li, l in enumerate(layer_params_list):
                this_regrets = []
                this_run_times = []
                this_min_vals = []
                this_min_val_inds = []
                this_relu_folds = [None for ki in enumerate(kfolds)]
                this_dnl_folds = [None for ki in enumerate(kfolds)]
                this_regression = [None for ki in enumerate(kfolds)]
                if li > 0:
                    for pi in param_range:
                        number_of_folds = 0

                        for ki, k in enumerate(kfolds):
                            this_df = df[mi][ji][li][pi][ki]

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
                    # Process Regression Folds
                    for ki, this_regression_fold in enumerate(this_regression):

                        regression_min_val_ind = np.argmin(this_regression_fold[:, RELU_VAL_REGRET_IND])
                        regression_min_regret = np.array(
                            this_regression_fold[regression_min_val_ind, RELU_REGRET_RATIO_IND])
                        if ki == 0:
                            regression_regrets = regression_min_regret
                        else:
                            regression_regrets = np.vstack((regression_regrets, regression_min_regret))

                    this_str = ["{}\%".format(str(abs(round(np.mean(regression_regrets), 2))))]
                    row_str.extend(this_str)
                    tabulate_df[mi][ji].extend(this_str)

                    this_regrets.insert(0, np.mean(regression_regrets))
                    this_run_times.insert(0, float("inf"))

                    # Process Relu Folds
                    for ki, this_relu_fold in enumerate(this_relu_folds):
                        min_val_ind = np.argmin(this_relu_fold[:, RELU_VAL_REGRET_IND])
                        min_val = this_relu_fold[min_val_ind, RELU_VAL_REGRET_IND]
                        min_regret = np.array(this_relu_fold[min_val_ind, RELU_REGRET_RATIO_IND])
                        run_time = np.array(this_relu_fold[min_val_ind, RELU_RUN_TIME_IND])

                        if ki == 0:
                            min_regrets = min_regret
                            run_times = run_time
                            min_vals = min_val
                            min_val_inds = min_val_ind
                        else:
                            min_regrets = np.vstack((min_regrets, min_regret))
                            run_times = np.vstack((run_times, run_time))
                            min_vals = np.vstack((min_vals, min_val))
                            min_val_inds = np.vstack((min_val_inds, min_val_ind))
                            # print(min_regret)
                    this_str = ["{}\%".format(str(abs(round(np.mean(min_regrets), 2))))]
                    tabulate_df[mi][ji].extend(this_str)
                    row_str.extend(this_str)
                    this_regrets.append(np.mean(min_regrets))
                    this_run_times.append(np.mean(run_times))

                    # Process SPO
                    if df_relu_spo is not None:
                        for ki, this_spo_fold in enumerate(df_relu_spo[mi][ji][li - 1]):
                            this_spo_fold = np.array(this_spo_fold.data).astype(float)
                            min_val_ind = np.argmin(this_spo_fold[:, SPO_VAL_REGRET_IND])
                            min_val = this_spo_fold[min_val_ind, SPO_VAL_REGRET_IND]
                            min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND]) * 100 / this_relu_fold[
                                0, RELU_OBJECTIVE_IND]
                            # min_regret = np.array(this_spo_fold[min_val_ind, SPO_REGRET_IND])
                            run_time = np.array(this_spo_fold[min_val_ind, SPO_RUN_TIME_IND])

                            if ki == 0:
                                min_regrets = min_regret
                                run_times = run_time
                                min_vals = min_val
                                min_val_inds = min_val_ind
                            else:
                                min_regrets = np.vstack((min_regrets, min_regret))
                                run_times = np.vstack((run_times, run_time))
                                min_vals = np.vstack((min_vals, min_val))
                                min_val_inds = np.vstack((min_val_inds, min_val_ind))
                                # print(min_regret)
                        this_str = ["{}\%".format(str(abs(round(np.mean(min_regrets), 2))))]

                        tabulate_df[mi][ji].extend(this_str)
                        row_str.extend(this_str)
                        this_regrets.append(np.mean(min_regrets))
                        this_run_times.append(np.mean(run_times))

                    max_regret_ind_str = np.argmax(np.array(this_regrets))
                    min_runtime_ind_str = np.argmin(np.array(this_run_times))

                    tabulate_df[mi][ji][(li - 1) * 3 + max_regret_ind_str + 1] = "\\textbf{{{}}}".format(
                        tabulate_df[mi][ji][(li - 1) * 3 + max_regret_ind_str + 1])
                    row_str[(li - 1) * 3 + max_regret_ind_str + 1] = row_str[
                                                                         (li - 1) * 3 + max_regret_ind_str + 1] + "*"
                    row_str[(li - 1) * 3 + min_runtime_ind_str + 1] = row_str[
                                                                          (li - 1) * 3 + min_runtime_ind_str + 1] + "^"
                    myTable.add_row(row_str)
    print(myTable)
    for mi, m in enumerate(machines):
        print(tabulate(np.array(tabulate_df[mi]), tablefmt="latex_raw", floatfmt=".2f"))


def print_all_regrets_table(df, machines, jobs, layer_params_list, param_range, kfolds):
    first_row = ["Jobs", "M1 Reg", "M1 R D10", "M1 R D5 5", "M1 R D1",
                 "M2 Reg", "M2 R D10", "M2 R D5 5", "M2 R D1",
                 "M3 Reg", "M3 R D10", "M3 R D5 5", "M3 R D1",
                 "M4 Reg", "M4 R D10", "M4 R D5 5", "M4 R D1"]
    myTable = PrettyTable(first_row)

    tabulate_df = [[None for j in jobs] for m in machines]
    row_index = 0
    for mi, m in enumerate(machines):
        machine_row = ["******M{}******".format(m) for i in range(len(first_row))]
        myTable.add_row(machine_row)
        for ji, j in enumerate(jobs):
            row_str = [str(ji + 1)]
            tabulate_df[mi][ji] = [str(ji + 1)]
            this_regrets = []
            this_run_times = []
            row_index = 0
            for li, l in enumerate(layer_params_list):
                this_regression = [None for ki in enumerate(kfolds)]
                for pi in param_range:
                    number_of_folds = 0

                    for ki, k in enumerate(kfolds):
                        this_df = df[mi][ji][li][pi][ki]

                        if this_df is not None:
                            number_of_folds += 1

                            this_fold = np.array(this_df.data).astype(float)
                            if this_regression[ki] is None:
                                this_regression[ki] = this_fold[this_fold[:, 1] == 0]
                            else:
                                this_regression[ki] = np.vstack((this_regression[ki], this_fold[this_fold[:, 1] == 0]))

                            min_val_ind = np.argmin(this_fold[:, RELU_VAL_REGRET_IND])
                            min_regret = np.array(this_fold[min_val_ind, RELU_REGRET_RATIO_IND])
                            run_time = np.array(this_fold[min_val_ind, RELU_RUN_TIME_IND])

                            if number_of_folds == 1:
                                min_regrets = min_regret
                                run_times = run_time
                            else:
                                min_regrets = np.vstack((min_regrets, min_regret))
                                run_times = np.vstack((run_times, run_time))

                            # print(min_regret)
                    this_regrets.append(np.mean(min_regrets))
                    this_run_times.append(np.mean(run_times))
                    # this_str = ["{} ({})".format(str(abs(round(np.mean(min_regrets), 2))),
                    #                               str(round(np.mean(run_times), 3)))]

                    this_str = ["{}".format(str(abs(round(np.mean(min_regrets), 2))))]
                    if number_of_folds != 5:
                        this_str[0] += "!"
                    row_str.extend(this_str)
                    tabulate_df[mi][ji].extend(this_str)

                # Process Regression

                for ki, this_regression_fold in enumerate(this_regression):

                    regression_min_val_ind = np.argmin(this_regression_fold[:, RELU_VAL_REGRET_IND])
                    regression_min_regret = np.array(
                        this_regression_fold[regression_min_val_ind, RELU_REGRET_RATIO_IND])
                    if ki == 0:
                        regression_regrets = regression_min_regret
                    else:
                        regression_regrets = np.vstack((regression_regrets, regression_min_regret))

                this_regrets.insert(li * 4, np.mean(regression_regrets))
                this_run_times.insert(li * 4, float("inf"))
                row_str.insert(li * 4 + 1, "{}".format(str(abs(round(np.mean(regression_regrets), 2)))))
                tabulate_df[mi][ji].insert(li * 4 + 1, "{}".format(str(abs(round(np.mean(regression_regrets), 2)))))
            max_regret_ind_str = np.argmax(np.array(this_regrets))
            min_runtime_ind_str = np.argmin(np.array(this_run_times))

            row_str[max_regret_ind_str + 1] = row_str[max_regret_ind_str + 1] + "*"
            tabulate_df[mi][ji][max_regret_ind_str + 1] = tabulate_df[mi][ji][max_regret_ind_str + 1] + "*"
            row_str[min_runtime_ind_str + 1] = row_str[min_runtime_ind_str + 1] + "^"
            myTable.add_row(row_str)

    print(myTable)
    for mi, m in enumerate(machines):
        print(tabulate(np.array(tabulate_df[mi]), tablefmt="latex_raw", floatfmt=".2f"))


def plot_relu_table(is_spo=True):
    kfolds = [0, 1, 2, 3, 4]
    loads = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 400, 40, 41, 42, 43, 44, 45, 46, 47, 48, 500, 501, 50, 51, 52, 53,
             54, 55, 56, 57]
    jobs = [i + 1 for i in range(10)]
    machines = [i + 1 for i in range(3)]
    param_range = [0, 1, 2]
    models_list = ['ReLuDNL']
    layer_params_list = ['01', "0501", "0901", "050501"]
    layer_params_list_spo = layer_params_list[1:-1]
    file_name_list = [[[[generate_file_name_ReLu_scheduling(load=loads[((m - 1) * len(jobs) + j - 1)], fold=kfold,
                                                            layer_params=layer_params) for kfold in kfolds] for
                        layer_params in layer_params_list] for j in jobs] for m in machines]

    df = combine_data_scheduling(file_name_list, machines, jobs, layer_params_list, kfolds)

    if is_spo:
        file_name_list_spo = [[[generate_file_name_SPO_scheduling(load=loads[((m - 1) * len(jobs) + j - 1)], fold=kfold)
                                for kfold in kfolds] for j in jobs] for m in machines]

        df_spo = get_spo_df(file_name_list_spo, machines, jobs, kfolds)

        file_name_list_relu_spo = [
            [[[generate_file_name_non_linear_SPO_scheduling(load=loads[((m - 1) * len(jobs) + j - 1)], fold=kfold,
                                                            layer_params=layer_params) for kfold in kfolds] for
              layer_params in layer_params_list[1:]] for j in jobs] for m in machines]

        df_nl_spo = get_non_linear_spo_df(file_name_list_relu_spo, machines, jobs, layer_params_list[1:], kfolds)

    file_name_list_intopt = [[[generate_file_name_intopt_scheduling(load=loads[((m - 1) * len(jobs) + j - 1)], fold=kfold)
                                for kfold in kfolds] for j in jobs] for m in machines]
    df_intopt = get_intopt_df(file_name_list_intopt, machines, jobs, kfolds)

    # print_all_regrets_table(df=df, machines=machines, jobs=jobs, layer_params_list=layer_params_list, param_range=param_range, kfolds=kfolds)

    # print_nonlinearity_table(df=df, machines=machines, jobs=jobs, layer_params_list=layer_params_list,
    #                         param_range=param_range, kfolds=kfolds)

    print_simple_regrets_table(df=df, machines=machines, jobs=jobs, layer_params_list=layer_params_list,
                            param_range=param_range, kfolds=kfolds, df_relu_spo=df_nl_spo, df_spo = df_spo, df_intopt=df_intopt)
    # #
    # print_simple_nonlinearity_vs_regression_table(df=df, machines=machines, jobs=jobs, layer_params_list=layer_params_list,
    #                         param_range=param_range, kfolds=kfolds, df_spo = df_spo, df_relu_spo = df_nl_spo      )
    # #
    # print_divider_table(df=df, machines=machines, jobs=jobs, layer_params_list=layer_params_list,
    #                     param_range=param_range, kfolds=kfolds)


if __name__ == "__main__":
    plot_relu_table()
    # parse_file_ReLu()
