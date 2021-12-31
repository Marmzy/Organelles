#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import random
import torch
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import Dataset


class OrganelleDataset(Dataset):
    def __init__(self, X_path, y_path, transform):
        self.images = pd.read_csv(X_path, header=None)
        self.transform = transform

        #Encoding the labels
        labels = pd.read_csv(y_path, header=None)
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(labels[0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.images.values[idx][0])
        img = img.convert("I")
        img = self.transform(img)

        label = self.labels[idx]
        file = self.images.iloc[idx, :][0].split("/")[-1].split(".")[0]
        return img, label, file
