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


def train_model(X, y, n_threads):
    # Set the range of hyperparameters to search
    C_range = np.logspace(-3, 3, 7)
    gamma_range = np.logspace(-3, 3, 7)
    # Set the grid of hyperparameters to search
    param_grid = {'C': C_range, 'gamma': gamma_log_range}
    # Use a cross-validated grid search to find the optimal hyperparameters
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=10, n_jobs=n_threads)
    clf.fit(X, y)
    # Return the trained model
    return clf


def main(input_file, output_file, n_threads):
    # Read the input data
    data = np.loadtxt(input_file, delimiter='\t')
    X = data[:, 1:]
    y = data[:, 0]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale the input data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Train the model
    clf = train_model(X_train, y_train, n_threads)
    # Evaluate the model on the testing data
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # Save the trained model to a file
    joblib.dump(clf, output_file)


if __name__ == '__main__':
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_file', type=str, required=True, help='Input file containing the data in TSV format')
    parser.add_argument('-out', '--output_file', type=str, required=True, help='Output file to store the trained model')
    parser.add_argument('-n', '--n_threads', type=int, default=1, help='Number of threads to use for parallel processing')
    args = parser.parse_args()

    # Run the main function
    main(args.input_file, args.output_file, args.n_threads)
