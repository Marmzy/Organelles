#!/usr/bin/env bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage(){
    if [ -n "$1" ]; then
        echo -e "${RED}→ $1${CLEAR}"
    fi
    echo "Usage: $0 [-h help] [-v verbose] [-d data] [-a arch] [-l lr] [-e epochs] [-b batch] [-m metric] [-k kfold]"
    echo " -h, --help       Print this help and exit"
    echo " -v, --verbose    Print verbose messages"
    echo " -d, --data       Name of the data directory"
    echo " -a, --arch       Model architecture"
    echo " -l, --lr         ADAM learning rate"
    echo " -e, --epochs     Number of epochs"
    echo " -b, --batch      Minibatch size"
    echo " -m, --metric     Evaluation metric (accuracy | sensitivity | specificity | precision | f1)"
    echo " -k, --kfold      Number of folds the training dataset was split into"
    echo ""
    echo "Example: $0 -d data -a vgg16_bn -l 0.00001 -e 10 -b 8 -m accuracy -k 5"
    exit 1
}

#Parsing command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -v|--verbose) VERBOSE=true ;;
        -d|--data) INPUT="$2"; shift ;;
        -a|--arch) ARCH="$2"; shift ;;
        -l|--lr) LR="$2"; shift ;;
        -e|--epochs) EPOCHS="$2"; shift ;;
        -b|--batch) BATCH="$2"; shift ;;
        -m|--metric) METRIC="$2"; shift ;;
        -k|--kfold) KFOLD="$2"; shift ;;
    esac
    shift
done

#Verifying arguments
if [ -z "$VERBOSE" ]; then VALUE_V=false; else VALUE_V=true; fi;
if [ -z "$INPUT" ]; then usage "Input directory name is not specified"; else IN_DIR=$INPUT; fi;
if [ -z "$ARCH" ]; then usage "Model architecture is not specified"; else VALUE_A=$ARCH; fi;
if [ -z "$LR" ]; then usage "Learning rate if not specified"; else VALUE_R=$LR; fi
if [ -z "$EPOCHS" ]; then usage "Number of epochs is not specified"; else VALUE_E=$EPOCHS; fi;
if [ -z "$BATCH" ]; then usage "Minibatch size is not specified"; else VALUE_B=$BATCH; fi;
if [ -z "$METRIC" ]; then usage "Evaluation metric is not specified"; else VALUE_M=$METRIC; fi;
if [ -z "$KFOLD" ]; then usage "Number of kfolds is not specified"; else VALUE_K=$KFOLD; fi;


#Asserting the directory and output subdirectory exists
if [ ! -d ${PWD%/*}/$IN_DIR ]; then
    echo "The specified directory name '${IN_DIR}' does not exist in ${PWD%/*}"
    exit 1
elif [ ! -e ${PWD%/*}/$IN_DIR/output ]; then
    mkdir -p ${PWD%/*}/$IN_DIR/output
fi

#Training the model
python3 ${PWD%/*}/src/data_train.py \
        --verbose $VALUE_V \
        --data $IN_DIR \
        --model $VALUE_A \
        --lr $VALUE_R \
        --epochs $VALUE_E \
        --batch $VALUE_B \
        --metric $VALUE_M \
        --kfold $VALUE_K
