from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import argparse
import threading
import re
import bioutils

import seaborn as sns
import matplotlib.pyplot as plt

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-seq", "--sequences_file", required=True, help="Input FASTA file with protein sequences")
parser.add_argument("-ann", "--annotations_file", required=True, help="Input TSV file with annotations for each protein")
parser.add_argument("-o", "--output_file", required=True, help="Output TSV file to save the classified sequences")
parser.add_argument("-m", "--model_file", required=True, help="File containing the trained random forest model")
parser.add_argument("-t", "--threshold", default=0.5, help="Probability threshold for high confidence predictions")
parser.add_argument("-T", "--n_threads", default=1, help="Number of threads to use for classification")
args = parser.parse_args()

# Read the input FASTA file with protein sequences
sequences = bioutils.read_fasta(args.sequences_file)

# Read the input TSV file with annotations for each protein
annotations = pd.read_csv(args.annotations_file, sep="\t")

# Join the annotations and protein sequences into a single data frame
df = pd.concat([annotations, sequences], axis=1)

# Load the trained random forest model from a file
with open(args.model_file, "rb") as f:
    clf = pickle.load(f)

# Classify the protein sequences using the trained random forest model
predictions = clf.predict(df.drop(["id", "sequence"], axis=1))
probabilities = clf.predict_proba(df.drop(["id", "sequence"], axis=1))

# Save the classified sequences to an output TSV file
with open(args.output_file, "w") as f:
    f.write("id\tsequence\tclass\tprobability\n")
    for i in range(len(predictions)):
        f.write("{}\t{}\t{}\t{:.6f}\n".format(df["id"][i], df["sequence"][i], predictions[i], probabilities[i][1]))

# Save the sequences classified with high confidence to a separate file
with open(args.output_file + ".high_confidence.tsv", "w") as f:
    f.write("id\tsequence\tclass\tprobability\n")
    for i in range(len(predictions)):
        if probabilities[i][1] >= args.threshold:
            f.write("{}\t{}\t{}\t{:.6f}\n".format(df["id"][i], df["sequence"][i], predictions[i], probabilities[i][1]))

# Save the sequences classified with high confidence to a separate FASTA file
with open(args.output_file + ".high_confidence.fasta", "w") as f:
    for i in range(len(predictions)):
        if probabilities[i][1] >= args.threshold:
            f.write(">{}\n{}\n".format(df["id"][i], df["sequence"][i]))