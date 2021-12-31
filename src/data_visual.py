#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

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


def main():

    #Parse arguments from the command line
    args = parseArgs()

    #Initialising variables
    path = os.path.join("/".join(os.getcwd().split("/")[:-1]))
    fout = os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_predictions.txt")
    fout2 = os.path.join(path, args.indir, "output", args.infiles, args.infiles + "_images.png")

    #Creating evaluation images
    output_img(fout, fout2, path)


if __name__ == '__main__':
    main()
