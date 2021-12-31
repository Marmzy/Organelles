# Organelles

![vgg16_bn image](https://github.com/Marmzy/Organelles/blob/main/data/output/vgg16_bn_weighted_lr1e-05_decay0.0_epochs10_batch8_f1/vgg16_bn_weighted_lr1e-05_decay0.0_epochs10_batch8_f1_images.png)

This project came about when I found the [2D HeLa](https://ome.grc.nia.nih.gov/iicbu2008/hela/index.html) dataset <sup>1</sup>.
The dataset contains grayscale fluorescence microscopy images of HeLa cells stained with various organelle-specific fluorescent dyes.
The dataset contains images of 10 organelles, thus in this project I shall attempt to train a model to classify this images correctly.
According to the National Institute of Aging website, the highest published performance for the 2D HeLa dataset is 95.3% <sup>2</sup>.

For this project I will start by using pre-existing trained CNNs. If the predictions are subpar, I shall try to create and train a new NN.

This project was made on Ubuntu 18.04 using Windows Subsystem for Linux 2.

## Starting off

To run the code, a suitable development environment must be set up. The GNU/Linux environment needs to have Python3 (>3.8.1).
Ideally a virtual environment is setup using Anaconda or pyenv to ensure this project don't interfere with any other.

All scripts necessary for the analysis can be found in the 'scripts' directory.

## Preprocessing

[`01_dataprep.sh`](https://github.com/Marmzy/Organelles/blob/main/scripts/01_dataprep.sh) will download the 2D HeLa dataset and setup the data/output directory structure.

```bash
Usage: 01_dataprep.sh [-h help] [-v verbose] [-o output] [-t test] [-k kfold]
 -h, --help       Print this help and exit
 -v, --verbose    Print verbose messages
 -o, --output     Name of output directory where data will be stored
 -t, --test       Size of the test dataset
 -k, --kfold      Number of folds to split the training dataset into

Example: 01_dataprep.sh -o data -t 0.2 -k 5
```

The dataset will be split into temporary train and test, and the temporary train dataset will further be split into k train and validation datasets.
In the end, I want to train k models and evaluate these all on the same test dataset.

## Training

[`02_model_train_eval.sh`](https://github.com/Marmzy/Organelles/blob/main/scripts/02_model_train_eval.sh) will train a [pre-existing model](https://pytorch.org/vision/stable/models.html)
and save the model on the epoch it achieves the highest metric score on the validation dataset.

```bash
Usage: 02_model_train_eval.sh [-h help] [-v verbose] [-d data] [-a arch] [-l lr] [-e epochs] [-b batch] [-m metric] [-k kfold]
 -h, --help       Print this help and exit
 -v, --verbose    Print verbose messages
 -d, --data       Name of the data directory
 -a, --arch       Model architecture
 -l, --lr         ADAM learning rate
 -e, --epochs     Number of epochs
 -b, --batch      Minibatch size
 -m, --metric     Evaluation metric (accuracy | sensitivity | specificity | precision | f1)
 -k, --kfold      Number of folds the training dataset was split into

Example: 02_model_train_eval.sh -d data -a vgg16_bn -l 0.00001 -e 10 -b 8 -m f1 -k 5
```

The training process goes as follows:
For each dataset,
 - the dataset average and standard deviations are calculated
 - images are transformed by resizing, flipping and rotating the images and normalising them
 - the models are then tuned using Adam
 - the model is saved at the epoch when it achieves the highest score on the validation dataset given a specified metric
 
## Evaluation

[`03_make_pred.sh`](https://github.com/Marmzy/Organelles/blob/main/scripts/03_make_pred.sh) will score the k models on the same test dataset and outputs a figure
showing the top 5 best and worst predictions **across all models** .

```bash
Usage: 03_make_pred.sh [-h help] [-v verbose] [-d data] [- n name]
 -h, --help       Print this help and exit
 -v, --verbose    Print verbose messages
 -d, --data       Name of the data directory
 -n, --name       Name of the directory containing the models to evaluate

Example: 03_make_pred.sh -d data -n vgg16_bn_weighted_lr1e-05_decay0.0_epochs10_batch8_f1
```

Below are results I obtained by training (k=5) GoogleNet models:

| Accuracy | Sensitivity | Specificity | Precision | F1-Score |
| --- | --- | --- | --- | --- |
| 0.99792 | 0.98916 | 0.99884 | 0.99035 | 0.99035 |

![googlenet image](https://github.com/Marmzy/Organelles/blob/main/data/output/googlenet_weighted_lr0.0001_decay0.0_epochs10_batch8_f1/googlenet_weighted_lr0.0001_decay0.0_epochs10_batch8_f1_images.png)

An explanation of the image above:
 - organelle_num is the original name of the image on which a model was evaluated
 - truth denotes the correct organelle
 - pred denotes the predicted organelle, together with %confidence of the correct label
 
Example:
- In the image above, image "ER_085" was correctly predicted to be a fluorescence microscopy image of the ER. The model predicted the organelle as "ER" with 99.98% confidence.
- Conversely, image "lysosome_074" was predicted as a fluorescence microscopy image of the endosome, whilst in actuality being an image of the lysosome.
The model predicted the image to be of a lysosome with only 2.75% confidence.


---

<sup>1</sup>: M. V. Boland and R. F. Murphy (2001). A Neural Network Classifier Capable of Recognizing the Patterns of all Major Subcellular Structures in Fluorescence Microscope Images of HeLa Cells. Bioinformatics 17:1213-1223

<sup>2</sup>: A. Chebira, Y. Barbotin, C. Jackson, T. Merryman, G. Srinivasa, R.F. Murphy, and J. Kovacevic (2007). A multiresolution approach to automated classification of protein subcellular location images. BMC Bioinformatics 8:210
