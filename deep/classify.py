import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# Set the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, help="Path to the trained model")
parser.add_argument("-s", "--sequences", type=str, required=True, help="Path to the protein amino acid sequences file in fasta format")
parser.add_argument("-p", "--probability_threshold", type=float, default=0.5, help="Minimum probability threshold for saving sequences")
parser.add_argument("-o", "--output_prefix", type=str, default="output", help="Prefix for the output files")
parser.add_argument("-t", "--num_threads", type=int, default=1, help="Number of threads to use for data loading and preprocessing")
args = parser.parse_args()

# Load the protein amino acid sequences from the fasta file
sequences = []
sequence_ids = []
current_id = None
current_sequence = []
with open(args.sequences) as f:
  for line in f:
    if line.startswith(">"):
      if current_id:
        sequences.append("".join(current_sequence))
        current_sequence = []
      current_id = line.strip()[1:]
      sequence_ids.append(current_id)
    else:
      current_sequence.append(line.strip())
sequences.append("".join(current_sequence))

# Create the TensorFlow session
with tf.Session() as sess:
  # Load the trained model
  saver = tf.train.import_meta_graph(args.model + ".meta")
  saver.restore(sess, args.model)

  # Get the input and output tensors
  input_sequence = tf.get_default_graph().get_tensor_by_name("input_sequence:0")
  input_secondary_structure = tf.get_default_graph().get_tensor_by_name("input_secondary_structure:0")
  input_solvent_accessibility = tf.get_default_graph().get_tensor_by_name("input_solvent_accessibility:0")
  input_contact_map = tf.get_default_graph().get_tensor_by_name("input_contact_map:0")
  logits = tf.get_default_graph().get_tensor_by_name("logits:0")

  # Create a data generator to load and augment the data
  def data_generator():
    while True:
      for sequence in sequences:
        yield load_data(sequence)

# Create a data iterator to iterate over the data generator
iterator = tf.data.Iterator.from_structure(tf.data.get_output_types(data_generator()), tf.data.get_output_shapes(data_generator()))
next_element = iterator.get_next()
iterator_init_op = iterator.make_initializer(tf.data.Dataset.from_generator(data_generator, output_types=tf.data.get_output_types(data_generator()), output_shapes=tf.data.get_output_shapes(data_generator())))

# Set the number of threads for data loading and preprocessing
num_threads = args.num_threads if args.num_threads > 0 else 1
tf.data.experimental.set_num_parallel_calls(iterator_init_op, num_threads)

# Initialize the data iterator
sess.run(iterator_init_op)

# Make predictions using the trained model
predictions = []
while True:
  try:
    # Get the next batch of data
    X = sess.run(next_element)

    # Make predictions using the trained model
    batch_predictions = sess.run(logits, feed_dict={input_sequence: X[0], input_secondary_structure: X[1], input_solvent_accessibility: X[2], input_contact_map: X[3]})
    predictions.extend(batch_predictions)
  except tf.errors.OutOfRangeError:
    # End the prediction loop when the data iterator is exhausted
    break

# Create the output dataframe
output_df = pd.DataFrame(predictions, columns=["Class1", "Class2", "Class3", "Class4", "Class5"])
output_df["SequenceID"] = sequence_ids
output_df["Sequence"] = sequences

# Save the output dataframe as a tsv file
output_df.to_csv(args.output_prefix + "_predictions.tsv", sep="\t", index=False)

# Create a dataframe with the high confidence predictions
high_confidence_df = output_df[output_df.max(axis=1) >= args.probability_threshold]

# Save the high confidence dataframe as a tsv file
high_confidence_df.to_csv(args.output_prefix + "_high_confidence.tsv", sep="\t", index=False)

# Save the high confidence sequences as a fasta file
with open(args.output_prefix + "_high_confidence.fasta", "w") as f:
  for index, row in high_confidence_df.iterrows():
    f.write(">{}\n".format(row["SequenceID"]))
    f.write(row["Sequence"] + "\n")
