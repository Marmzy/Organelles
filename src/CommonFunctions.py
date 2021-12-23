#!/usr/bin/env python

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from CustomImageDataset import OrganelleDataset
from torch.utils.data import DataLoader


def confusion_matrix(truths, preds):

    #Initialising variables
    df_list = []

    #Looping over all the classes
    for c in range(10):
        tp, fp, fn, tn = 0, 0, 0, 0
        for t, p in zip(truths, preds):
            if t == c and p == c:
                tp += 1
            elif t != c and p == c:
                fp += 1
            elif t == c and p != c:
                fn += 1
            else:
                tn += 1

        #Calculating several metrics
        accuracy = (tp + tn) / (tp + fp + fn + tn)

        if tp + fn > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = np.nan

        if tn + fp > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = np.nan

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = np.nan

        if not np.isnan(sensitivity) and not np.isnan(specificity) and not np.isnan(precision):
            if precision + sensitivity > 0:
                f1 = 2 * (( precision * sensitivity) / (precision + sensitivity))
            else:
                f1 = np.nan
        else:
            f1 = np.nan

        df_list.append(pd.DataFrame({"ACC": [accuracy], "SEN": [sensitivity], "SPE": [specificity], "PRE": [precision],"F1": [f1]}))

    #Merging the metric dataframes
    df = pd.concat(df_list, ignore_index=True)
    macro_f1 = 2 * ((df["PRE"].mean() * df["PRE"].mean()) / (df["PRE"].mean() + df["PRE"].mean()))

    #Returning metrics
    return df["ACC"].mean(), df["SEN"].mean(), df["SPE"].mean(), df["PRE"].mean(), macro_f1


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
