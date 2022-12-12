# Import the necessary libraries
import argparse
import multiprocessing
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from Bio import SeqIO
from Bio.PDB import PDBList, PDBIO
from Bio.PDB.Polypeptide import three_to_one
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def process_sequence(sequence, clf):
    # Predict the secondary structure, solvent accessibility, and contact map for the protein
    secondary_structure = chou_fasman(sequence)
    solvent_accessibility = dssp(sequence)
    contact_map = psicov(sequence)
    # Compute the feature vector for the protein
    X = np.concatenate((secondary_structure, solvent_accessibility, contact_map))
    X = X.reshape(1, -1)
    # Make a prediction using the trained model
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)
    return (sequence, y_pred, y_prob)


def main(model_file, fasta_file, output_file, sequence_ids_file, n_threads):
    # Load the trained model from a file
    clf = joblib.load(model_file)
    # Read the FASTA file containing the protein sequences
    sequences = [str(seq.seq) for seq in SeqIO.parse(fasta_file, 'fasta')]
    # Use multiple threads to process the protein sequences in parallel
    pool = multiprocessing.Pool(processes=n_threads)
    results = pool.map(process_sequence, sequences)
    pool.close()
    pool.join()
    # Write the results to the output files
    with open(output_file, 'w') as f:
        f.write('\t'.join(['sequence_id', 'prediction', 'probability']) + '\n')
        for result in results:
            f.write('\t'.join(map(str, result)) + '\n')
    with open(sequence_ids_file, 'w') as f:
        for result in results:
            if result[2][result[1]] >= 0.9:
                f.write(result[0] + '\n')


if __name__ == '__main__':
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', type=str, required=True, help='Input file containing the trained model')
    parser.add_argument('-f', '--fasta_file', type=str, required=True, help='Input file containing the protein sequences in FASTA format')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output file to store the predictions')
    parser.add_argument('-s', '--sequence_ids_file', type=str, required=True, help='Output file to store the sequence IDs of proteins with high confidence predictions')
    parser.add_argument('-n', '--n_threads', type=int, default=1, help='Number of threads to use for parallel processing')
    args = parser.parse_args()

    # Run the main function
    main(args.model_file, args.fasta_file, args.output_file, args.sequence_ids_file, args.n_threads)