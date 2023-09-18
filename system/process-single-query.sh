#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./process-single-query <pathToQuery>"
fi

python3 single-query.py $1 | python3 filter.py