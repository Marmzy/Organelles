#!/usr/bin/env python

import argparse
import glob
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from collections import Counter, deque
from CommonFunctions import confusion_matrix, get_image_mean
from CustomImageDataset import OrganelleDataset
from torch.utils.data import DataLoader


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def parseArgs():

    parser = argparse.ArgumentParser(description='Evaluating models on the test dataset')

    #Options for input and output
    parser.add_argument('--indir', type=str, help='Name of the data directory')
    parser.add_argument('--infiles', type=str, help='Stem name of the input files')
    parser.add_argument('--kfold', type=int, help='Number of cross-validation folds to split the training dataset into')
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False, help='Print verbose messages')

    #Printing arguments to the command line
    args = parser.parse_args()

    print('Called with args:')
    print(args)

    return args


def get_weights(data, path, device, verbose):

    #Initialising variables
    suffix = ""

    #Loading the data
    with open(os.path.join(path, data, "test", "y_test.txt")) as f:
        y_train = [line.rstrip() for line in f]
    count_dict = Counter(y_train)

    #Checking for class imbalance
    if verbose:
        print("\nChecking for class imbalance...\n")

        #Looping over the classes
        for key, item in sorted(count_dict.items()):
            print("Number of {} images: {}".format(key, str(item)))
        print("\t=> Very slight data imbalance\n")

    #Calculate weights to alleviate the small class imbalance
    weights = [1 - (int(v) / sum(count_dict.values())) for k, v in sorted(count_dict.items())]
    return torch.FloatTensor(weights).to(device)


def make_preds(model, loss, data_loader, k, fout, device, verbose):

    #Writing the predictions to a file
    with open(fout, "a") as f:
        print("Evaluating the model trained on training dataset fold {}...".format(k))

        #Setting the data and model
        data = data_loader["test"]
        model.train(False)

        #Looping over the minibatches
        for idx, (data_test, target_test) in enumerate(data):
            x, y = data_test.to(device), target_test.to(device)

            with torch.set_grad_enabled(False):
                y_pred = model(x)
                prob = nn.functional.softmax(y_pred, dim=1)
                pred = y_pred.cpu().argmax(dim=1)
                l = loss(y_pred, y)

                #Saving the predictions
                for i in range(len(pred)):
                    print("{}\t{}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}".format(
                        pred[i].item(), y[i].item(), prob[i][0], prob[i][1],  prob[i][2], prob[i][3], prob[i][4], prob[i][5], prob[i][6], prob[i][7], prob[i][8], prob[i][9]), file=f)


def score(fout, fout2):

    #Writing the evaluation metrics to this file
    with open(fout2, 'w') as f:

        #Loading the predictions data
        df = pd.read_csv(fout, sep="\t")
        truths = df.loc[:, "GroundTruth"].to_numpy()
        preds = df.loc[:, "Prediction"].to_numpy()

        #Calculating various metrics
        acc, sen, spe, pre, f1 = confusion_matrix(truths, preds)

        #Outputting the metrics
        with open(fout, "a") as fout2:
            print("\nModel evaluation metrics:")
            print("\nModel evaluation metrics:", file=f)
            print("Accuracy\tSensitivity\tSpecificity\tPrecision\tF1-Score")
            print("Accuracy\tSensitivity\tSpecificity\tPrecision\tF1-Score", file=f)
            print("{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}".format(acc, sen, spe, pre, f1))
            print("{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}\t\t{:.5f}".format(acc, sen, spe, pre, f1), file=f)


def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialising variables
    path = os.path.join("/".join(os.getcwd().split("/")[:-1]))
    test_dir = os.path.join(path, args.indir, "test")
    fout = os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_predictions.txt")
    fout2 = os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_evaluation.txt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Writing the header to the output file
    with open(fout, 'w') as f:
        f.write("Prediction\tGroundTruth\tProbActin\tProbDNA1\tProbEndosome\tProbER\tProbGolgia\tProbGolgpp\tProbLysosome\tProbMicrotubules\tProbMitochondria\tProbNucleolus\n")

    #Getting the mean and standard deviation of our dataset
    img_mean, img_std = get_image_mean(os.path.join(path, args.indir, "test/X_test.txt"), os.path.join(path, args.indir, "test/y_test.txt"), int(args.infiles.split("_")[6].split("batch")[1]))

    #Defining image transformation techniques to apply
    image_transform = {
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: torch.from_numpy(np.array(img).astype(np.float32)).unsqueeze(0)),
            transforms.Normalize((img_mean), (img_std)),
            transforms.Lambda(lambda image: image.expand(3, -1, -1))
        ])
    }

    #Loading the test dataset
    if args.verbose:
        print("Loading the test dataset...")
    organelle_test = OrganelleDataset(os.path.join(path, args.indir, "test/X_test.txt"), os.path.join(path, args.indir, "test/y_test.txt"), image_transform["test"])
    test_data = DataLoader(organelle_test, batch_size=int(args.infiles.split("_")[6].split("batch")[1]), shuffle=True, pin_memory=True)
    data_loader = {"test": test_data}

    #Looping over the K folds
    for k in range(args.kfold):

        #Initialising the model
        trained_model = glob.glob(os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_fold{}*.pkl".format(k)))[0]
        if args.verbose:
            print("Loading model: {}...\n".format(os.path.basename(trained_model)))
        model_arch = args.infiles.split("_weighted")[0]

        #Setting up the model
        model = models.__dict__[model_arch](pretrained=True)
        if "vgg" in model_arch:
            model.classifier[6] = nn.Linear(4096, 10)
        elif "resnet" in model_arch:
            num_feat = model.fc.in_features
            model.fc = nn.Linear(num_feat, 10)
        elif "google" in model_arch:
            model.fc = nn.Linear(1024, 10)

        model = model.to(device)
        model.load_state_dict(torch.load(trained_model), strict=False)

        #Settings for training the model
        weights = get_weights(args.indir, path, device, args.verbose)
        loss = nn.CrossEntropyLoss(weight=weights)

        #Making the predictions
        make_preds(model, loss, data_loader, k, fout, device, args.verbose)

    #Scoring the evaluations
    score(fout, fout2)



if __name__ == '__main__':
    main()
