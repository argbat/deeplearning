# Para corre, bajarse glove.6B.100d.txt y colocarlo en datasets

import nltk
import argparse

import numpy as np
import os
import pandas as pd
import tensorflow as tf

from gensim import corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from itertools import zip_longest
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

def read_args():
    parser = argparse.ArgumentParser(
        description='Training a MLP on the petfinder dataset')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    parser.add_argument('--dataset_dir', default='./datasets', type=str,
                        help='Directory with the training and test files.')
    parser.add_argument('--hidden_layer_sizes', nargs='+', default=[100], type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--dropouts', nargs='+', default=[0.5], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of instances in each batch.')
    parser.add_argument('--experiment_name', type=str, default='Base model',
                        help='Name of the experiment, used in mlflow.')
    parser.add_argument('--one_hot_columns', nargs='+', type=str, default=['Gender', 'Color1'],
                        help='Name of column to be one hot encoded.')
    parser.add_argument('--embedding_columns', nargs='+', type=str, default=['Breed1'],
                        help='Name of columns to be embedded.')
    parser.add_argument('--filter_widths', nargs='+', default=[2, 3, 5], type=int,
                        help='Filters widths.')
    parser.add_argument('--filter_count', type=int, default=64,
                        help='Filter count.')

    args = parser.parse_args()

    assert len(args.hidden_layer_sizes) == len(args.dropouts)
    return args

args = read_args()

nltk.download(["punkt", "stopwords"]);

dataset = pd.read_csv(os.path.join('./datasets/', 'train.csv'))

target_col = 'AdoptionSpeed'
nlabels = dataset[target_col].unique().shape[0]

SW = set(stopwords.words("english"))

def tokenize_description(description):
    return [w.lower() for w in word_tokenize(description, language="english") if w.lower() not in SW]

# Fill the null values with the empty string to avoid errors with NLTK tokenization
dataset["TokenizedDescription"] = dataset["Description"].fillna(value="").apply(tokenize_description)

vocabulary = corpora.Dictionary(dataset["TokenizedDescription"])
vocabulary.filter_extremes(no_below=1, no_above=1.0, keep_n=10000)

embeddings_index = {}

with open("./datasets/glove.6B.100d.txt", "r") as fh:
    for line in fh:
        values = line.split()
        word = values[0]
        if word in vocabulary.token2id:  # Only use the embeddings of words in our vocabulary
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

# It's important to always use the same one-hot length
one_hot_columns = {
    one_hot_col: dataset[one_hot_col].max()
    for one_hot_col in args.one_hot_columns
    # ['Gender', 'Color1', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health']
}
embedded_columns = {
    embedded_col: dataset[embedded_col].max() + 1
    for embedded_col in args.embedding_columns
    # ['Breed1']
}
numeric_columns = ['Age', 'Fee']

def dataset_generator(ds, test_data=False):
    for _, row in ds.iterrows():
        instance = {}
        
        # One hot encoded features
        instance["direct_features"] = [
            tf.keras.utils.to_categorical(row[one_hot_col] - 1, max_value)
            for one_hot_col, max_value in one_hot_columns.items()
        ]

        # Numeric features (should be normalized beforehand)
        # for numeric_column in numeric_columns:
        #     instance["direct_features"].append(tf.keras.utils.normalize(row[numeric_column]))

        instance["direct_features"] = np.hstack(instance["direct_features"])

        # Embedded features
        for embedded_col in embedded_columns:
            instance[embedded_col] = [row[embedded_col]]
        
        # Document to indices for text data, truncated at MAX_SEQUENCE_LEN words
        instance["description"] = vocabulary.doc2idx(
            row["TokenizedDescription"],
            unknown_word_index=len(vocabulary)
        )[:MAX_SEQUENCE_LEN]
        
        # One hot encoded target for categorical crossentropy
        if not test_data:
            target = tf.keras.utils.to_categorical(row[target_col], nlabels)
            yield instance, target
        else:
            yield instance

# Set output types of the generator (for numeric types check the type is valid)
instance_types = {
    "direct_features": tf.float32,
    "description": tf.int32
}

for embedded_col in embedded_columns:
    instance_types[embedded_col] = tf.int32
        
tf_dataset = tf.data.Dataset.from_generator(
    lambda: dataset_generator(dataset),
    output_types=(instance_types, tf.int32)
)

TRAIN_SIZE = int(dataset.shape[0] * 0.8)
DEV_SIZE = dataset.shape[0] - TRAIN_SIZE
BATCH_SIZE = args.batch_size

shuffled_dataset = tf_dataset.shuffle(TRAIN_SIZE + DEV_SIZE, seed=42)

# Pad the datasets to the max value for all the "non sequence" features
padding_shapes = (
    {k: [-1] for k in ["direct_features"] + list(embedded_columns.keys())},
    [-1]
)

MAX_SEQUENCE_LEN = 55

# Pad to MAX_SEQUENCE_LEN for sequence features
padding_shapes[0]["description"] = [MAX_SEQUENCE_LEN]

# Pad values are irrelevant for non padded data
padding_values = (
    {k: 0 for k in list(embedded_columns.keys())},
    0
)

# Padding value for direct features should be a float
padding_values[0]["direct_features"] = np.float32(0)

# Padding value for sequential features is the vocabulary length + 1
padding_values[0]["description"] = len(vocabulary) + 1

train_dataset = shuffled_dataset.skip(DEV_SIZE)\
    .padded_batch(BATCH_SIZE, padded_shapes=padding_shapes, padding_values=padding_values)

dev_dataset = shuffled_dataset.take(DEV_SIZE)\
    .padded_batch(BATCH_SIZE, padded_shapes=padding_shapes, padding_values=padding_values)

EMBEDDINGS_DIM = 100  # Given by the model (in this case glove.6B.100d)

embedding_matrix = np.zeros((len(vocabulary) + 2, 100))

for widx, word in vocabulary.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[widx] = embedding_vector
    else:
        # Random normal initialization for words without embeddings
        embedding_matrix[widx] = np.random.normal(size=(100,))  

# Random normal initialization for unknown words
embedding_matrix[len(vocabulary)] = np.random.normal(size=(100,))

tf.keras.backend.clear_session()

# Add one input and one embedding for each embedded column
embedding_layers = []
inputs = []
for embedded_col, max_value in embedded_columns.items():
    input_layer = tf.keras.layers.Input(shape=(1,), name=embedded_col)
    inputs.append(input_layer)
    # Define the embedding layer
    embedding_size = int(max_value / 4)
    embedding_layers.append(
        tf.squeeze(
            tf.keras.layers.Embedding(
                input_dim=max_value, 
                output_dim=embedding_size
            )(input_layer), 
            axis=-2
        )
    )
    print('Adding embedding of size {} for layer {}'.format(embedding_size, embedded_col))

# Add the direct features already calculated
direct_features_input = tf.keras.layers.Input(
    shape=(sum(one_hot_columns.values()),), 
    name='direct_features'
)
inputs.append(direct_features_input)

# Word embedding layer
description_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LEN,), name="description")
inputs.append(description_input)

word_embeddings_layer = tf.keras.layers.Embedding(
    embedding_matrix.shape[0],
    EMBEDDINGS_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LEN,
    trainable=False,
    name="word_embedding"
)(description_input)

description_features = []
for filter_width in args.filter_widths:
    layer = tf.keras.layers.Conv1D(
        args.filter_count,
        filter_width,
        activation="relu",
        name="conv_{}_words".format(filter_width)
    )(word_embeddings_layer)
    layer = tf.keras.layers.GlobalMaxPooling1D(name="max_pool_{}_words".format(filter_width))(layer)
    description_features.append(layer)

## TODO: Create a NN (CNN or RNN) for the description input (replace the next)
feature_map = tf.keras.layers.Concatenate(name="feature_map")(
    description_features + [direct_features_input] + embedding_layers
)

if len(args.hidden_layer_sizes) > 0:
    hidden_layer_size = args.hidden_layer_sizes.pop(0)
    last_hidden_layer = tf.keras.layers.Dense(hidden_layer_size, activation='relu')(feature_map)
if len(args.dropouts) > 0:
    dropout = args.dropouts.pop(0)
    last_hidden_layer = tf.keras.layers.Dropout(dropout)(last_hidden_layer)
    
if len(args.dropouts) > len(args.hidden_layer_sizes):
    args.dropouts = args.dropouts[:len(args.hidden_layer_sizes)]
    
for hidden_layer_size, dropout in zip_longest(args.hidden_layer_sizes, args.dropouts, fillvalue=None):
    if hidden_layer_size != None:
        last_hidden_layer = tf.keras.layers.Dense(hidden_layer_size, activation='relu')(last_hidden_layer)
    if dropout != None:
        last_hidden_layer = tf.keras.layers.Dropout(dropout)(last_hidden_layer)
    
output_layer = tf.keras.layers.Dense(nlabels, activation='softmax')(last_hidden_layer)

model = tf.keras.models.Model(inputs=inputs, outputs=[output_layer], name="amazing_model")

model.compile(loss='categorical_crossentropy', 
              optimizer='nadam',
              metrics=['accuracy'])
model.summary()

import mlflow

mlflow.set_experiment('awesome_advanced_approach')

with mlflow.start_run(nested=True):
    # Log model hiperparameters first
    mlflow.log_param('filter_widths', args.filter_widths)
    mlflow.log_param('filter_count', args.filter_count)
    mlflow.log_param('hidden_layer_sizes', args.hidden_layer_sizes)
    mlflow.log_param('dropouts', args.dropouts)
    mlflow.log_param('embedding_columns', args.embedding_columns)
    mlflow.log_param('one_hot_columns', args.one_hot_columns)
    # mlflow.log_param('numeric_columns', numeric_columns)
    
    # Train
    epochs = args.epochs
    history = model.fit(train_dataset, epochs=epochs, validation_data=dev_dataset, verbose=0)
    
    # TODO: analyze history to see if model converges/overfits
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.show()

    print("*** Dev loss: {} - accuracy: {}".format(history.history['val_loss'][-1], 
        history.history['val_accuracy'][-1]))

    mlflow.log_metric('dev_loss', history.history['val_loss'][-1])
    mlflow.log_metric('dev_accuracy', history.history['val_accuracy'][-1])
    mlflow.log_metric('epochs', epochs)
    mlflow.log_metric('train_loss', history.history['loss'][-1])
    mlflow.log_metric('train_accuracy', history.history['accuracy'][-1])

test_dataset = pd.read_csv(os.path.join('./datasets/', 'test.csv'))

# First tokenize the description

test_dataset["TokenizedDescription"] = test_dataset["Description"]\
    .fillna(value="").apply(tokenize_description)

# Generate the basic TF dataset

tf_test_dataset = tf.data.Dataset.from_generator(
    lambda: dataset_generator(test_dataset, True),
    output_types=instance_types  # It should have the same instance types
)

test_data = tf_test_dataset.padded_batch(
    BATCH_SIZE, 
    padded_shapes=padding_shapes[0], 
    padding_values=padding_values[0]
)

test_dataset[target_col] = model.predict(test_data).argmax(axis=1)

test_dataset.to_csv("./submission.csv", index=False, columns=["PID", "AdoptionSpeed"])

