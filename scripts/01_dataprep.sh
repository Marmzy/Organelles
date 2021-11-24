#!/usr/bin/env bash

CLEAR='\033[0m'
RED='\033[0;31m'

function usage(){
    if [ -n "$1" ]; then
        echo -e "${RED}→ $1${CLEAR}"
    fi
    echo "Usage: $0 [-h help] [-v verbose] [-o output] [-t test] [-k kfold]"
    echo " -h, --help       Print this help and exit"
    echo " -v, --verbose    Print verbose messages"
    echo " -o, --output     Name of output directory where data will be stored"
    echo " -t, --test       Size of the test dataset"
    echo " -k, --kfold      Number of folds to split the training dataset into"
    echo ""
    echo "Example: $0 -o data -t 0.2 -k 5"
    exit 1
}

#Parsing command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage ;;
        -v|--verbose) VERBOSE=true ;;
        -o|--output) OUTPUT="$2"; shift ;;
        -t|--test) TEST="$2"; shift ;;
        -k|--kfold) KFOLD="$2"; shift ;;
    esac
    shift
done

#Verifying arguments
if [ -z "$VERBOSE" ]; then VALUE_V=false; else VALUE_V=true; fi;
if [ -z "$OUTPUT" ]; then OUT_DIR="data"; else OUT_DIR=$OUTPUT; fi;
if [ -z "$TEST" ]; then VALUE_T=0.2; else VALUE_T=$TEST; fi;
if [ -z "$KFOLD" ]; then usage "Number of kfolds is not specified"; else VALUE_K=$KFOLD; fi;


#Making the output directory and subsequent subdirectories if it doesn't exist yet
if [ ! -e ${OUT_DIR} ]; then
    mkdir -p ${PWD%/*}/${OUT_DIR}
    mkdir -p ${PWD%/*}/${OUT_DIR}/raw
    mkdir -p ${PWD%/*}/${OUT_DIR}/train
    mkdir -p ${PWD%/*}/${OUT_DIR}/val
    mkdir -p ${PWD%/*}/${OUT_DIR}/test
fi

#Downloading the raw data if it hasn't been done so before
if [ -z "$(ls -A ${PWD%/*}/${OUT_DIR}/raw)" ]; then
    wget -P ${PWD%/*}/${OUT_DIR}/raw "https://ome.grc.nia.nih.gov/iicbu2008/hela.tar.gz"
    tar -zxf ${PWD%/*}/${OUT_DIR}/raw/hela.tar.gz -C ${PWD%/*}/${OUT_DIR}/raw
#    rm ${PWD%/*}/${OUT_DIR}/raw/hela.tar.gz
fi

#Preparing the data
python3 ${PWD%/*}/src/data_prep.py \
        --verbose ${VALUE_V} \
        --output ${OUT_DIR} \
        --test ${VALUE_T} \
        --kfold ${VALUE_K}
