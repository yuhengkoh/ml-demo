#!/bin/bash

#PBS -N test
#PBS -q gpu
#PBS -l ncpus=3:ngpus=12:mem=24GB
#PBS -o test.log
#PBS -e error.log

module load anaconda3
python embedder2.py