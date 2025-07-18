#!/bin/bash

#PBS -N test
#PBS -q normal
#PBS -l select=1:ncpus=1:mem=24GB
#PBS -o test.log
#PBS -e error.log

python embedder.py