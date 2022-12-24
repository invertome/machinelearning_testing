#!/bin/bash

# update conda
conda update -n base -c defaults conda

# create a new conda environment
conda create -n MLtests python=3.7

# activate the new environment
conda activate MLtests

# add the necessary channels
conda config --add channels anaconda
conda config --add channels bioconda
conda config --add channels conda-forge

# install the dependencies
conda install -c anaconda argparse
conda install -c anaconda ipython
conda install -c anaconda numpy
conda install -c anaconda scikit-learn
conda install -c anaconda scipy
conda install -c bioconda biopython
conda install -c anaconda pandas
conda install -c anaconda seaborn
conda install -c anaconda matplotlib
conda install -c conda-forge tensorflow
