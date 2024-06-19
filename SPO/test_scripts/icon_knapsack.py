import argparse
import sys
import os
import inspect



currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandparentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, grandparentdir)
from Experiments import test_knapsack_SPO

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("layer_params", metavar='Layer Parameters', type=int,
                        help="list of layer parameters for the model. Enter as a string. Ex: 1 5,1 9,1 5,5,1",
                        nargs='+')
    args = parser.parse_args()

    layer_params = [[9] + args.layer_params]

    kfolds = [0, 1, 2, 3, 4]
    n_iter = 10
    dest_folder = grandparentdir+'/Tests/Icon_knapsack/SPORelax/'
    capacities = [12, 24, 48, 72, 96, 120, 144, 172, 196, 220]

    for n in range(n_iter):
        for capacity in capacities:
            test_knapsack_SPO(capacity=capacity, is_shuffle=False, NUMBER_OF_RANDOM_TESTS=1, kfolds=kfolds, n_iter=1,
                              dest_folder=dest_folder, layer_params = layer_params, noise_level=0)

if __name__ == "__main__":
    main()


