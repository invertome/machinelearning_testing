#!/bin/bash


# create a new conda environment
conda create -n MLtests python=3.7 -y

# activate the new environment
conda activate MLtests

# add the necessary channels
conda config --add channels anaconda
conda config --add channels bioconda
conda config --add channels conda-forge

# install the dependencies
conda install -c anaconda argparse -y
conda install -c anaconda ipython -y
conda install -c anaconda numpy -y
conda install -c anaconda scikit-learn -y
conda install -c anaconda scipy -y
conda install -c bioconda biopython -y
conda install -c anaconda pandas -y
conda install -c anaconda seaborn -y
conda install -c anaconda matplotlib -y
conda install -c conda-forge tensorflow -y
