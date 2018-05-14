import pandas as pd
import tensorflow as tf
import numpy as np

TRAIN_URL = "C:\\Users\mponsser\Desktop\ML project\ga_data_training.csv"
TEST_URL = "C:\\Users\mponsser\Desktop\ML project\ga_data_test.csv"

CSV_COLUMN_NAMES = ['Transaccion', 'Localizacion', 'Superficie', 'Habitaciones', 'Banos', 'PrecioCompra', 'PrecioAlquiler', 'Outputs']
#CSV_COLUMN_NAMES = ['Transaccion', 'Localizacion', 'Superficie', 'Habitaciones', 'Banos', 'Precio', 'Outputs']
OUTPUTS = ['Url 0', 'Url 1', 'Url 2', 'Url 3', 'Url 4', 'Url 5', 'Url 6', 'Url 7', 'Url 8', 'Url 9', 'Url 10', 'Url 11']

 # Create a local copy of the training set
def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def load_data(label_name='Outputs'):
    """Returns the dataset as (train_features, train_label), (test_features, test_label)."""
    train_path, test_path = maybe_download()

   # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path, names=CSV_COLUMN_NAMES, header=0, encoding='latin-1')
    
    # Train now holds a pandas DataFrame, which is data structure analogous to a table

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0, encoding='latin-1')
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames
    return (train_features, train_label), (test_features, test_label)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset