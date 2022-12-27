# Import the necessary libraries
import argparse
import numpy as np
import joblib
import multiprocessing
import rpy2.robjects as ro
import os
import subprocess
import tempfile
from sklearn.svm import SVC
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.PDB import PDBList, PDBIO
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.DSSP import DSSP
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Install the bio3d package if it is not already installed
ro.r("if (!require('bio3d')) install.packages('bio3d')")
# Load the bio3d package
ro.r("library('bio3d')")

def dssp(sequence):
    # Predict the 3D structure of the protein using the bio3d package
    pdb = ro.r['bio3d::predict.structure'](sequence, method='rosetta')
    # Create a temporary PDB file for the model
    pdb_file = "temp.pdb"
    with open(pdb_file, "w") as f:
        f.write(pdb)
    # Create a DSSP object
    dssp = DSSP(pdb_file, "temp")
    # Get the solvent accessibility predictions for each residue in the protein
    solvent_accessibility = []
    for residue in dssp:
        # residue is a tuple with the following elements:
        # (residue id, residue name, solvent accessibility prediction, etc.)
        solvent_accessibility.append(residue[3])
    return solvent_accessibility


def get_secondary_structure(sequence):
    # Conver Seq object into string
    sequence_str = str(sequence)
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write the sequence to a temporary FASTA file
        fasta_path = os.path.join(tmpdir, "temp.fasta")
        with open(fasta_path, "w") as fasta_file:
            fasta_file.write(">temp\n")
            fasta_file.write(sequence_str)
        # Call the run_model.py script and pass the temporary FASTA file as input
        result = subprocess.run(["python", "run_model.py", "--device", "gpu", "--outfmt", "fas", fasta_path], capture_output=True)
        # Extract the predicted secondary structure from the output
        secondary_structure = result.stdout.decode().strip()
    return secondary_structure



def process_sequence(sequence):
    # Predict the secondary structure, solvent accessibility, and contact map for the protein
    # Predict the secondary structure of the protein sequence
    secondary_structure = get_secondary_structure(sequence)
    solvent_accessibility = dssp(sequence)
    contact_map = ro.r['bio3d::predict.contact'](sequence, method='psipred')
    return (sequence, secondary_structure, solvent_accessibility, contact_map)


def main(fasta_file, output_file, n_threads):
    # Parse the FASTA file
    sequences = []
    for record in SeqIO.parse(fasta_file, 'fasta'):
        sequences.append(record.seq)
    # Process the sequences in parallel using multiple threads
    pool = multiprocessing.Pool(n_threads)
    results = pool.map(process_sequence, sequences)
    pool.close()
    pool.join()
    # Write the results to the output file
    with open(output_file, 'w') as out_f:
        for result in results:
            out_f.write('{}\t{}\t{}\t{}\n'.format(result[0], result[1], result[2], result[3]))


if __name__ == '__main__':
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--fasta_file', type=str, required=True, help='Input file containing the protein sequences in FASTA format')
    parser.add_argument('-out', '--output_file', type=str, required=True, help='Output file to store the results in TSV format')
    parser.add_argument('-n', '--n_threads', type=int, default=1, help='Number of threads to use for parallel processing')
    args = parser.parse_args()

    # Run the main function
    main(args.fasta_file, args.output_file, args.n_threads)
