 #!/usr/bin/env python

import argparse
import copy
import numpy as np
import os
import pandas as pd
import pickle as pkl
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from collections import Counter, deque
from CommonFunctions import confusion_matrix, get_image_mean
from CustomImageDataset import OrganelleDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    parser = argparse.ArgumentParser(description='Downloading and preprocessing the data')

    #Options for input and output
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False, help='Print verbose messages')
    parser.add_argument('--data', type=str, help='Path to the data directory')
    parser.add_argument('--kfold', type=int, help='Number of cross-validation folds to split the training dataset into')

    #Options for the optimizer
    parser.add_argument('--decay', type=float, default=0.0, nargs='?', help='ADAM weight decay (default: 0.0)')
    parser.add_argument('--lr', type=float, help='ADAM gradient descent optimizer learning rate')

    #Options for training
    parser.add_argument('--model', type=str, help='Model architecture')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch', type=int, help='Minibatch size')
    parser.add_argument('--metric', type=str, default='accuracy', help='Evaluation metric of the model (default: accuracy)')

    #Printing arguments to the command line
    args = parser.parse_args()

    print('Called with args:')
    print(args)

    return args


def get_weights(data, path, k, device, verbose):

    #Initialising variables
    suffix = ""

    #Loading the data
    with open(os.path.join(path, data, "train", "y_train_{}.txt".format(k))) as f:
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


def train_model(model, loss, optimizer, epochs, data_loader, k, fout, device, metric, verbose):

    #Initialising variables
    pkl_queue = deque()
    best_metric = -1.0
    best_loss = 100.0
    best_model_weights = model.state_dict()
    since = time.time()
    end = time.time()


    with open(fout + ".log", "w") as f:
        if verbose:
            print(model, "\n")
        print(model, "\n", file=f)

        print("Analysing cross-validation fold {}...".format(k))
        print("Analysing cross-validation fold {}...".format(k), file=f)

        #Looping over the epochs
        for epoch in range(epochs):
            print("Epoch:{}/{}".format(epoch+1, epochs), end="")
            print("Epoch:{}/{}".format(epoch+1, epochs), end="", file=f)

            #Making sure training and validating occurs seperately
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train(True)
                else:
                    model.train(False)

                #Setting the data
                data = data_loader[phase]

                #Initialising more variables
                running_loss = 0
                truths, preds = [], []

                #Looping over the minibatches
                for idx, (data_train, target_train) in enumerate(data):
                    optimizer.zero_grad()
                    x, y = data_train.to(device), target_train.to(device)

                    with torch.set_grad_enabled(phase == "train"):
                        y_pred = model(x)
                        l = loss(y_pred, y)

                        pred = y_pred.cpu().argmax(dim=1)
                        pred = pred.reshape((len(pred), 1))
                        preds.append(pred)

                        truth = target_train.numpy()
                        truth = truth.reshape((len(truth), 1))
                        truths.append(truth)

                        if phase == "train":
                            l.backward()
                            optimizer.step()

                    #Calculating statistics
                    running_loss += l.item()

                #Scoring predictions through various metrics
                truths, preds = np.vstack(truths), np.vstack(preds)
                acc, sens, spef, prec, f1 = confusion_matrix(truths, preds)
                epoch_loss = running_loss / len(x)

                print("\t{} Loss: {:.4f} Acc: {:.4f} Sens: {:.4f} Spef: {:.4f} Prec: {:.4f} F1: {:.4f} Time: {:.4f}".format(phase, epoch_loss, acc, sens, spef, prec, f1, time.time()-end), end="")
                print("\t{} Loss: {:.4f} Acc: {:.4f} Sens: {:.4f} Spef: {:.4f} Prec: {:.4f} F1: {:.4f} Time: {:.4f}".format(phase, epoch_loss, acc, sens, spef, prec, f1, time.time()-end), end="", file=f)

                #Saving the model with the highest target metric for the validation data
                if phase == "val":
                    print("\n", end="")
                    print("\n", end="", file=f)
                    if metric == "accuracy":
                        model_score = acc
                        m = "acc"
                    elif metric == "sensitivity":
                        model_score = sens
                        m = "sens"
                    elif metric == "specificity":
                        model_score = spef
                        m = "spef"
                    elif metric == "precision":
                        model_score = prec
                        m = "prec"
                    elif metric == "f1":
                        model_score = f1
                        m = "f1"

                    if model_score > best_metric:
                        best_metric = model_score
                        best_loss = epoch_loss
                        best_model_weights = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), "{}_epoch{}.pkl".format(fout, epoch+1))
                        pkl_queue.append("{}_epoch{}.pkl".format(fout, epoch+1))
                        if len(pkl_queue) > 1:
                            pkl_file = pkl_queue.popleft()
                            os.remove(pkl_file)

                end = time.time()

        #Print overall training information
        time_elapsed = time.time() - since
        print("\nTraining completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print("\nTraining completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60), file=f)
        print("Best val {}: {:.4f}".format(m, best_metric))
        print("Best val {}: {:.4f}".format(m, best_metric), file=f)


def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialising variables
    path = os.path.join("/".join(os.getcwd().split("/")[:-1]))
    fdir = os.path.join("/".join(os.getcwd().split("/")[:-1]), "data", "output")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Looping over the K folds
    for k in range(args.kfold):

        #Setting up the model
        model = models.__dict__[args.model](pretrained=True)
        if "vgg" in args.model:
            model.classifier[6] = nn.Linear(4096, 10)
        elif "resnet" in args.model:
            num_feat = model.fc.in_features
            model.fc = nn.Linear(num_feat, 10)
        elif "google" in args.model:
            model.fc = nn.Linear(1024, 10)

        model = model.to(device)

        #Getting the mean and standard deviation of our dataset
        img_mean, img_std = get_image_mean(os.path.join(path, "data/train/X_train_{}.txt".format(k)), os.path.join(path, "data/train/y_train_{}.txt".format(k)), args.batch)

        #Defining image transformation techniques to apply
        image_transform = {
            "train": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.Lambda(lambda img: torch.from_numpy(np.array(img).astype(np.float32)).unsqueeze(0)),
                transforms.Normalize((img_mean), (img_std)),
                transforms.Lambda(lambda image: image.expand(3, -1, -1))
            ]),
            "val": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda img: torch.from_numpy(np.array(img).astype(np.float32)).unsqueeze(0)),
                transforms.Normalize((img_mean), (img_std)),
                transforms.Lambda(lambda image: image.expand(3, -1, -1))
            ])
        }

        #Creating the training dataset
        if args.verbose:
            print("Loading train dataset {}...".format(k))
        organelle_train = OrganelleDataset(os.path.join(path, "data/train/X_train_{}.txt".format(k)), os.path.join(path, "data/train/y_train_{}.txt".format(k)), image_transform["train"])
        organelle_val = OrganelleDataset(os.path.join(path, "data/val/X_val_{}.txt".format(k)), os.path.join(path, "data/val/y_val_{}.txt".format(k)), image_transform["val"])

        train_data = DataLoader(organelle_train, batch_size=args.batch, shuffle=True, pin_memory=True)
        val_data = DataLoader(organelle_val, batch_size=args.batch, shuffle=True, pin_memory=True)

        data_loader = {"train": train_data, "val": val_data}

        #Creating the output dir/file name
        fname = "{}_weighted_lr{}_decay{}_epochs{}_batch{}_{}".format(args.model, args.lr, args.decay, args.epochs, args.batch, args.metric)

        #Creating the output directory if it does not yet exist
        if os.path.isdir(os.path.join(fdir, fname)):
            fout = os.path.join(fdir, fname, fname + "_fold{}".format(k))
        else:
            os.makedirs(os.path.join(fdir, fname))
            fout = os.path.join(fdir, fname, fname + "_fold{}".format(k))

        #Checking the data
        weights = get_weights(args.data, path, k, device, args.verbose)

        #Settings for training the model
        loss = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

        #Training the model
        train_model(model, loss, optimizer, args.epochs, data_loader, k, fout, device, args.metric, args.verbose)


if __name__ == '__main__':
    main()

