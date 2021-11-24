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


def flatten(l):
    return [item for sublist in l for item in sublist]


def overview(path):

    print("\nCounting the number of images for each organelle...")

    #Looping over the subdirectories and counting the number of images for each organelle
    for subdir in sorted(glob.glob(os.path.join(path, "raw") + "/*")):
        print("There are {} images for {}".format(str(len([img for img in glob.glob(subdir + "/*.[tT][iI][fF]")])), os.path.basename(subdir)))


def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialsing variables
    path = os.path.dirname(os.getcwd())
    image_list = []
    organelle_list = []

    #Getting an overview of the input data
    if args.verbose:
        print(path)
        overview(os.path.join(path, args.output))

    #Looping over the subdirectories and getting the full paths to the images
    for subdir in sorted(glob.glob(os.path.join(path, "data/raw/*"))):
        image_list.append([img for img in glob.glob(subdir + "/*.[tT][iI][fF]")])
        organelle_list.append([os.path.basename(subdir) for img in glob.glob(subdir + "/*.[tT][iI][fF]")])

    #Creating a dataframe form the image paths
    X = flatten(image_list)
    y = flatten(organelle_list)

    #Splitting the data into (temporary) train and test
    if args.verbose:
        print("\nSplitting the dataset and into train ({}%) and test ({}%)...".format(str(int((1-float(args.test))*100)), str(int(float(args.test)*100))))
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=args.test, stratify=y)

    #Splitting the temporary training dataset into K training and validation datasets
    if args.verbose:
        print("\nSplitting the temporary training dataset into {} folds...".format(args.kfold))

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True)

    for idx, (train_index, val_index) in enumerate(skf.split(X_temp, y_temp)):
        if args.verbose:
            print("TRAIN:", train_index, "VAL:", val_index)

        X_train, X_val = np.array(X_temp)[train_index], np.array(X_temp)[val_index]
        y_train, y_val = np.array(y_temp)[train_index], np.array(y_temp)[val_index]

        #Saving the train and validation datasets
        np.savetxt(os.path.join(path, "data", "train", "X_train_{}.txt".format(idx)), X_train, fmt='%s')
        np.savetxt(os.path.join(path, "data", "train", "y_train_{}.txt".format(idx)), y_train, fmt='%s')

        np.savetxt(os.path.join(path, "data", "val", "X_val_{}.txt".format(idx)), X_val, fmt='%s')
        np.savetxt(os.path.join(path, "data", "val", "y_val_{}.txt".format(idx)), y_val, fmt='%s')

    #Saving the test dataset
    np.savetxt(os.path.join(path, "data", "test", "X_test.txt"), X_test, fmt='%s')
    np.savetxt(os.path.join(path, "data", "test", "y_test.txt"), y_test, fmt='%s')

    if args.verbose:
        print("\nSaved the training datasets to: {}".format(os.path.join(path, "data", "train")))
        print("Saved the validation datasets to: {}".format(os.path.join(path, "data", "val")))
        print("Saved the test dataset to: {}".format(os.path.join(path, "data", "test")))


if __name__ == '__main__':
    main()
