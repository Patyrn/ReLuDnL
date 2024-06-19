import argparse
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandparentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, grandparentdir)

from Experiments import test_SPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("load", help="scheduling problem loads", type=int)
    parser.add_argument("layer_params", metavar='Layer Parameters', type=int,
                        help="list of layer parameters for the model. Enter as a string. Ex: 1 5,1 9,1 5,5,1",
                        nargs='+')
    args = parser.parse_args()
    load = args.load
    layer_params = [[9] + args.layer_params]

    if load == 1:
            loads = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    elif load == 3:
            loads = [500, 501, 50, 51, 52, 53, 54, 55, 56, 57]
    elif load == 2:
            loads = [400, 40, 41, 42, 43, 44, 45, 46, 47, 48]

    kfolds = [0, 1, 2, 3, 4]
    n_iter = 10
    dest_folder = grandparentdir+'/Tests/Icon_scheduling/SPORelax/'
    for n in range(n_iter):
        for this_load in loads:
            test_SPO(load=this_load, is_shuffle=True, layer_params=layer_params, kfolds=kfolds, n_iter=1,
                     dest_folder=dest_folder)


if __name__ == "__main__":
    # icon scheduling load divider layer params
    main()
