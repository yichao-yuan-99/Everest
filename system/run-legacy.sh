#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "./run-legacy.sh <pathToGraph> <pathToMotif> <delta> Baseline"
fi

python3 legacy.py $1 $2 $3 $4