import argparse
import glob
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parseArgs():

    parser = argparse.ArgumentParser(description='Downloading and preprocessing the data')

    #Options for input and output
    parser.add_argument('-o', '--output', type=str, help='Name of the output data directory')
    parser.add_argument('-v', '--verbose', type=str2bool, nargs='?', const=True, default=False, help='Print verbose messages')

    #Options for preprocessing
    parser.add_argument('--test', type=float, help='Ratio of samples that is included in the test dataset')
    parser.add_argument('--kfold', type=int, help='Number of cross-validation folds to split the training dataset into')

    #Printing arguments to the command line
    args = parser.parse_args()

    print('Called with args:')
    print(args)

    return args


def overview(path):

    print("\nCounting the number of images for each organelle...")

    #Looping over the subdirectories and counting the number of images for each organelle
    for subdir in sorted(glob.glob(os.path.join(path, "raw") + "/*")):
        print("There are {} images for {}".format(str(len([img for img in glob.glob(subdir + "/*.[tT][iI][fF]")])), os.path.basename(subdir)))




def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Getting an overview of the input data
    if args.verbose:
        overview(os.path.join(os.path.dirname(os.getcwd()), args.output))



if __name__ == '__main__':
    main()
