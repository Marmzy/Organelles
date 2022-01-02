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


def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialising variables
    path = os.path.join("/".join(os.getcwd().split("/")[:-1]))
    fout = os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_predictions.txt")
    fout2 = os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_images.png")
    fout3 = os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_scatter.png")

    #Creating evaluation images
    output_img(fout, fout2, path)

    #Creating a scatterplot of the predictions
    output_scatter(fout, fout3)


if __name__ == '__main__':
    main()
