#!/bin/bash

# Script to perform indexing and qrel step for retrieval system. needs parameter -i $input and -o $output

docker build --tag aramis-imarg:lastest ..

sh ./tira-run.sh -idx "$@"
sh ./tira-run.sh -qrel "$@"