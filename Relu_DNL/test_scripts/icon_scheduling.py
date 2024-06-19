import argparse
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandparentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, grandparentdir)

from Experiments import test_Icon_scheduling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("load", help="scheduling problem loads", type=int)
    parser.add_argument("divider", metavar="Heuristic Param Divider",
                        help='Decide the rate of heuristic parameter updates(all_params/divider)', type=int)
    parser.add_argument("layer_params", metavar='Layer Parameters', type=int,
                        help="list of layer parameters for the model. Enter as a string. Ex: 1 5,1 9,1 5,5,1",
                        nargs='+')
    args = parser.parse_args()
    load = args.load
    divider = args.divider
    layer_params = args.layer_params

    kfolds = [0, 1, 2, 3, 4]
    n_iter = 10
    file_folder = 'Tests/Icon_scheduling/ReLuDNL'
    for n in range(n_iter):
        test_Icon_scheduling(loads=[load], is_save=True, layer_params=layer_params, file_folder=file_folder, regression_epoch = 30,
                             params_per_epoch_divider=divider, test_boolean=[0, 1], n_iter=1, kfolds=kfolds,
                             dnl_epoch=10)


if __name__ == "__main__":
    # icon scheduling load divider layer params
    main()
