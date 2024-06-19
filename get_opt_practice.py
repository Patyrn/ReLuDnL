import argparse
import sys


def main(argv):

    parser = argparse.ArgumentParser()
    # parser.add_argument("ncapacities", help="Number of capacities to process", type=int)
    parser.add_argument("capacities", help="Capacity values for knapsacks. Enter as a string Ex: 12,24,48 ")
    parser.add_argument("layer_params", metavar='Layer Paramters', help="list of layer parameters for the model. Enter as a string. Ex: 1 5,1 9,1 5,5,1", nargs='+')
    args = parser.parse_args()
    capacities = args.capacities
    layer_params = args.layer_params
    for layer_param in layer_params:
        for c in capacities:
            print("layer params: {}, capacity: {}".format(layer_param,c))
if __name__ == "__main__":
   main(sys.argv[1:])