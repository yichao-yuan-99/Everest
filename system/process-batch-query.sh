#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: ./process-batch-query <graph> <gpu> <pathToQueryDir>"
fi

python3 batch-query.py $1 $2 $3 | python3 filter.py