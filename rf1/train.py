from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import argparse
import threading

import seaborn as sns
import matplotlib.pyplot as plt

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-in", "--input_file", required=True, help="Input TSV file with sequence annotations")
parser.add_argument("-o", "--output_file", required=True, help="Output file to save the trained model")
parser.add_argument("-t", "--test_size", default=0.2, help="Test set size as a proportion of the data")
parser.add_argument("-n", "--n_estimators", default=100, help="Number of trees in the random forest")
parser.add_argument("-T", "--n_threads", default=1, help="Number of threads to use for training")
args = parser.parse_args()

# Read the input TSV file
data = pd.read_csv(args.input_file, sep="\t")

# Split the data into training and test sets
X = data.drop(columns="class")
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)

# Train the random forest classifier in multiple threads
clf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=args.n_threads)
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)

# Save the test results to a file
with open("test_results.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.show()

# Save the trained model to a file
with open(args.output_file, "wb") as f:
    pickle.dump(clf, f)