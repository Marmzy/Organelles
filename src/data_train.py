#!/usr/bin/env python

import argparse
import copy
import numpy as np
import os
import pickle as pkl
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

#from CNN import ResNet34
from collections import deque
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


def get_image_mean(target_file, label_file, batch):

    #Defining image transformation techinques to apply
    transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.Lambda(lambda img: torch.from_numpy(np.array(img).astype(np.float32)).unsqueeze(0))
    ])

    #Creating an image dataset
    organelle_dataset = OrganelleDataset(target_file, label_file, transformations)
    image_loader = DataLoader(organelle_dataset, batch_size=batch, shuffle=False, pin_memory=True)

    #Looping over the minibatches and calculating the running mean and std
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
    for data, _ in image_loader:
        channels_sum += torch.mean(torch.Tensor.float(data), dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(torch.Tensor.float(data) ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def get_weights(data, path, normalise, denoise, k, verbose):

    #Initialising variable
    suffix = ""
    if normalise:
        suffix += "_normalised"
    elif denoise:
        suffix += "_denoised"

    #Loading the data
    y_train = np.load(os.path.join(path, data, "train", "y_train_{}.npy".format(k)))

    #Checking for class imbalance
    label0 = y_train.tolist().count(0)
    label1 = y_train.tolist().count(1)

    if verbose:
        print("\nChecking for class imbalance...\n")
        print("Number of Non-ectopic heartbeats: {}".format(label0))
        print("Number of Ventricular ectopic heartbeats: {}".format(label1))
        print("\t=> Dataset imbalance: ~= {:.2f}\n".format(label0 / label1))

    #Calculate weights to alleviate class imbalance
    weights = [1 - (samples / (label0 + label1)) for samples in [label0, label1]]
    return weights


def load_data(data, path, normalise, denoise, batch, k, train):

    #Initialising variable
    suffix = ""
    if normalise:
        suffix += "_normalised"
    elif denoise:
        suffix += "_denoised"

    #Constructing the PyTorch DataSet
    if train:
        train_data = ECGDataset(os.path.join(path, data, "train", "X_train{}_{}.npy".format(suffix, k)), os.path.join(path, data, "train", "y_train_{}.npy".format(k)))
        return DataLoader(train_data, batch_size=batch, shuffle=True)
    else:
        val_data = ECGDataset(os.path.join(path, data, "val", "X_val{}_{}.npy".format(suffix, k)), os.path.join(path, data, "val", "y_val_{}.npy".format(k)))
        return DataLoader(val_data, batch_size=batch, shuffle=True)


def train_model(model, loss, optimizer, epochs, data_loader, k, fout, device, metric, verbose):

    #Initialising variables
    pkl_queue = deque()
    best_acc = -1.0
    best_sens = -1.0
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
                running_correct = 0
                running_train_num = 0
                ventricular_correct = 0
                ventricular_size = 0

                #Looping over the minibatches
                for idx, (data_train, target_train) in enumerate(data):
                    optimizer.zero_grad()
                    x, y = data_train.to(device), target_train.to(device)

                    with torch.set_grad_enabled(phase == "train"):
                        y_pred = model(x)
                        _, predictions = torch.max(y_pred.data, 1)
                        l = loss(y_pred, y)

                        if phase == "train":
                            l.backward()
                            optimizer.step()

                    #Calculating statistics
                    running_loss += l.item()
                    running_correct += torch.eq(predictions, y).sum()
                    running_train_num += target_train.size(0)
                    ventricular_size += torch.sum(y)
                    ventricular_correct += torch.eq(y + predictions, torch.tensor([2]*len(y)).to(device)).sum().to(device)

                #Calculating mean statistics
                epoch_loss = running_loss / len(x)
                epoch_acc = running_correct.double() / running_train_num
                epoch_sens = 0.0
                if ventricular_size > 0.0:
                    epoch_sens = float(ventricular_correct) / float(ventricular_size)

                print("\t{} Loss: {:06.4f} Acc: {:06.4f} Sen: {:06.4f} Time: {:06.4f}".format(phase, epoch_loss, epoch_acc, epoch_sens, time.time()-end), end="")
                print("\t{} Loss: {:06.4f} Acc: {:06.4f} Sen: {:06.4f} Time: {:06.4f}".format(phase, epoch_loss, epoch_acc, epoch_sens, time.time()-end), end="", file=f)


                #Saving the best acc/sens model for the validation data
                if phase == "val":
                    print("\n", end="")
                    print("\n", end="", file=f)
                    if metric == "accuracy":
                        if (epoch_acc > best_acc) or (epoch_acc == best_acc and epoch_sens > best_sens):
                            best_sens = epoch_sens
                            best_acc = epoch_acc
                            best_loss = epoch_loss
                            best_model_weights = copy.deepcopy(model.state_dict())
                            torch.save(model.state_dict(), "{}_epoch{}.pkl".format(fout, epoch+1))
                            pkl_queue.append("{}_epoch{}.pkl".format(fout, epoch+1))
                            if len(pkl_queue) > 1:
                                pkl_file = pkl_queue.popleft()
                                os.remove(pkl_file)
                    elif metric == "sensitivity":
                        if (epoch_sens > best_sens) or (epoch_sens == best_sens and epoch_acc > best_acc):
                            best_sens = epoch_sens
                            best_acc = epoch_acc
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
        print("Best val acc: {:.4f}".format(best_acc))
        print("Best val acc: {:.4f}".format(best_acc), file=f)
        print("Best val sens: {:.4f}\n".format(best_sens))
        print("Best val sens: {:.4f}".format(best_sens), file=f)



def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialising variables
    path = os.path.join("/".join(os.getcwd().split("/")[:-1]))
    fdir = os.path.join("/".join(os.getcwd().split("/")[:-1]), "data", "output")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.__dict__[args.model](pretrained=True)

    #Looping over the K folds
    for k in range(args.kfold):

        #Getting the mean and standard deviation of our dataset
        img_mean, img_std = get_image_mean(os.path.join(path, "data/train/X_train_{}.txt".format(k)), os.path.join(path, "data/train/y_train_{}.txt".format(k)), args.batch)

        #Defining image transformation techinques to apply
        image_transform = {
            "train": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.Lambda(lambda img: torch.from_numpy(np.array(img).astype(np.float32)).unsqueeze(0)),
                transforms.Normalize((img_mean), (img_std))
            ])
        }

        #Creating the training dataset
        organelle_dataset = OrganelleDataset(os.path.join(path, "data/train/X_train_{}.txt".format(k)), os.path.join(path, "data/train/y_train_{}.txt".format(k)), image_transform["train"])
        image_loader = DataLoader(organelle_dataset, batch_size=args.batch, shuffle=False, pin_memory=True)

        #Creating the output dir/file name
#        if args.weight:
#            fname = "ResNet34_weighted{}_lr{}_decay{}_epochs{}_batch{}_{}".format(infix, args.lr, args.decay, args.epochs, args.batch, args.metric)

            #Creating the output directory if it does not yet exist
#            if os.path.isdir(os.path.join(fdir, fname)):
#                fout = os.path.join(fdir, fname, fname + "_fold{}".format(k))
#            else:
#                os.makedirs(os.path.join(fdir, fname))
#                fout = os.path.join(fdir, fname, fname + "_fold{}".format(k))

        #Checking the data
#        weights = get_weights(args.data, path, args.normalised, args.denoised, k, args.verbose)

        #Loading the data
#        if args.verbose:
#            print("Loading train dataset {}...".format(k))
#        train_data = load_data(args.data, path, args.normalised, args.denoised, args.batch, k, train=True)
#        val_data = load_data(args.data, path, args.normalised, args.denoised, args.batch, k, train=False)
#        data_loader = {"train": train_data, "val": val_data}

        #Initialising the model
#        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#        model = ResNet34().to(device)

        #Settings for training the model
#        weights = torch.FloatTensor(weights).to(device)
#        loss = nn.CrossEntropyLoss(weight=weights)
#        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

        #Training the model
#        train_model(model, loss, optimizer, args.epochs, data_loader, k, fout, device, args.metric, args.verbose)


if __name__ == '__main__':
    main()

