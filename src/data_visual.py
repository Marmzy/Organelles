#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from PIL import Image


def parseArgs():

    parser = argparse.ArgumentParser(description='Evaluating models on the test dataset')

    #Options for input and output
    parser.add_argument('--indir', type=str, help='Name of the data directory')
    parser.add_argument('--infiles', type=str, help='Stem name of the input files')
    parser.add_argument('--kfold', type=int, help='Number of cross-validation folds to split the training dataset into')

    #Printing arguments to the command line
    args = parser.parse_args()

    print('Called with args:')
    print(args)

    return args


def output_img(fout, fout2, path):

    #Initialising variable
    organelles = ["Actin", "DNA", "Endosome", "ER", "Golgia", "Golgpp", "Lysosome", "Microtubules", "Mitochondria", "Nucleolus"]

    #Loading the predictions file as dataframe
    df = pd.read_csv(fout, sep="\t")
    probabilities = [(idx, mini_df[1][mini_df[1]["GroundTruth"]+2]) for idx, mini_df in enumerate(df.iterrows())]
    pics = sorted(probabilities, key=lambda tup: tup[1])[-5:][::-1] + sorted(probabilities, key=lambda tup: tup[1])[:5][::-1]

    #Creating an image of the 5 best and worst predictions
    fig, ax = plt.subplots(2, 5, figsize=(25,10))
    for i, axi in enumerate(ax.flat):
        idx = pics[i][0]
        if "DNA" in df.iloc[idx, :]["Image"] or "ER" in df.iloc[idx, :]["Image"]:
            img = Image.open(os.path.join(path, "data/raw", df.iloc[idx, :]["Image"].split("_")[0].lower(), df.iloc[idx, :]["Image"] + ".TIF"))
        else:
            img = Image.open(os.path.join(path, "data/raw", df.iloc[idx, :]["Image"].split("_")[0].lower(), df.iloc[idx, :]["Image"] + ".tif"))
        pred = df.iloc[idx, :]["Prediction"]
        truth = df.iloc[idx, :]["GroundTruth"]

        axi.imshow(img, cmap='gray')
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_title("{}:\nTruth: {}  Pred: {}  ({})".format(df.iloc[idx, :]["Image"], organelles[truth], organelles[pred], pics[i][1]))

    plt.tight_layout()
    plt.savefig(fout2)


def output_scatter(fout, fout3):

    #Loading the predictions as a dataframe
    df = pd.read_csv(fout, sep="\t")

    #Initialising variables
    img_dict = {}
    dfs = []

    #Looping over the predictions
    for img in set(df["Image"].values):
        mini_df = df[df["Image"] == img]
        preds = mini_df.iloc[:, mini_df["GroundTruth"].values[0]+2]
        dfs.append(pd.DataFrame({"Preds": preds.values,
                                 "Image": mini_df["Image"].values,
                                 "Organelle": [list(mini_df.columns)[mini_df["GroundTruth"].values[0]+2][4:]]*5}))
        img_dict[img] = np.median(preds)

    #Creating a dataframe for the scatterplot and ordering the images from worst to best prediction
    order = [tup[0] for tup in sorted(img_dict.items(), key=lambda x: x[1])]
    concat_df = pd.concat(dfs)
    concat_df.Image = concat_df.Image.astype("category")
    concat_df["Image"] = concat_df["Image"].cat.set_categories(order)
    concat_df.sort_values("Image", inplace=True)

    #Creating a dataframe for the lineplot and ordering the images from worst to best prediction
    median_df = pd.DataFrame({"Image": img_dict.keys(), "Median": img_dict.values()})
    median_df.Image = median_df.Image.astype("category")
    median_df["Image"] = median_df["Image"].cat.set_categories(order)
    median_df.sort_values("Image", inplace=True)

    #Creating the image
    fig, ax = plt.subplots(1, figsize=(35,6))
    sns.scatterplot(data=concat_df, x="Image", y="Preds", hue="Organelle")
    sns.lineplot(data=median_df, x="Image", y="Median", linewidth=1, color="black")
    plt.xticks(rotation=50, ha="right")
    plt.xlabel("Test image name")
    plt.ylabel("Median confidence of correct organelle")
    plt.title("Median confidence score for correct organelle label on the test dataset")
    plt.tight_layout()
    plt.savefig(fout3)


def output_training(fout, fout4, k):

    #Initialising variables
    root = "/home/calvin/Projects/Organelles/data/output/"
    dir_name = fout.split("/")[-2]
    metrics_dict = {"accuracy": 5, "sensitivity": 7,
                    "specificity": 9, "precision": 11,
                    "f1": 13}

    #Finding the line at which the training statistics were output
    with open(os.path.join(os.path.dirname(fout), dir_name + "_fold0.log")) as f:
        for idx, line in enumerate(f.readlines()):
            if "Epoch" in line:
                line_start = idx
                break

    #Creating the figure
    fig, axs = plt.subplots(2, 5, figsize=(25,12))

    #Looping over the k models that were trained
    for i in range(k):
        df = pd.read_csv(os.path.join(os.path.dirname(fout), dir_name + "_fold{}.log".format(i)), sep=r"\s+", skiprows=line_start, header=None)
        metric = dir_name.split("_")[-1]
        max_epoch = df.iloc[:, metrics_dict[metric]+15].idxmax()

        axs[0, i].plot(range(1,11), df.iloc[:, 3][:-2].astype(float), linestyle='dashed', color="grey", label="Train")
        axs[0, i].plot(range(1,11), df.iloc[:, 18][:-2].astype(float), label="Validation")
        axs[0, i].legend()
        axs[0, i].axvline(max_epoch+1, linestyle='dashed', color="darkred")
        axs[0, i].title.set_text("Loss (model {})".format(str(i+1)))

        axs[1, i].plot(range(1,11), df.iloc[:, metrics_dict[metric]][:-2].astype(float), linestyle='dashed', color="grey", label="Train")
        axs[1, i].plot(range(1,11), df.iloc[:, metrics_dict[metric]+15][:-2].astype(float), label="Validation")
        axs[1, i].axvline(max_epoch+1, linestyle='dashed', color="darkred")
        axs[1, i].title.set_text("{} (model {})".format(metric.title(), str(i+1)))
        axs[1, i].legend()

    fig.suptitle("Training of {}".format(dir_name))
    plt.tight_layout()
    plt.savefig(fout4)


def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialising variables
    path = os.path.join("/".join(os.getcwd().split("/")[:-1]))
    fout = os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_predictions.txt")
    fout2 = os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_images.png")
    fout3 = os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_scatter.png")
    fout4 = os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_training.png")

    #Creating evaluation images
    output_img(fout, fout2, path)

    #Creating a scatterplot of the predictions
    output_scatter(fout, fout3)

    #Creating lineplots that reflect the training process
    output_training(fout, fout4, args.kfold)


if __name__ == '__main__':
    main()
