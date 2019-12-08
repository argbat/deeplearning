"""Exercise 1

Usage:

$ CUDA_VISIBLE_DEVICES=2 python practico_1_train_petfinder.py --dataset_dir ../ --epochs 30 --dropout 0.1 0.1 --hidden_layer_sizes 200 100

To know which GPU to use, you can check it with the command

$ nvidia-smi
"""

import argparse

import os
import mlflow
import numpy
import pandas
import tensorflow as tf
from itertools import zip_longest

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from sklearn.utils import shuffle

TARGET_COL = 'AdoptionSpeed'


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
    parser.add_argument('--dropout', nargs='+', default=[0.5], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of instances in each batch.')
    parser.add_argument('--experiment_name', type=str, default='Base model',
                        help='Name of the experiment, used in mlflow.')
    parser.add_argument('--one_hot_columns', nargs='+', type=str, default=['Gender', 'Color1'],
                        help='Name of column to be one hot encoded.')
    parser.add_argument('--embedding_columns', nargs='+', type=str, default=['Breed1'],
                        help='Name of columns to be embedded.')
    args = parser.parse_args()

    assert len(args.hidden_layer_sizes) == len(args.dropout)
    return args


def process_features(df, one_hot_columns, numeric_columns, embedded_columns, test=False):
    direct_features = []

    # Create one hot encodings
    for one_hot_col, max_value in one_hot_columns.items():
        direct_features.append(tf.keras.utils.to_categorical(df[one_hot_col] - 1, max_value))

    # TODO Create and append numeric columns
    # Don't forget to normalize!
    for numeric_column in numeric_columns:
        direct_features.append(tf.keras.utils.normalize(df[numeric_column].values.reshape(-1,1)))
    
    # Concatenate all features that don't need further embedding into a single matrix.
    features = {'direct_features': numpy.hstack(direct_features)}

    # Create embedding columns - nothing to do here. We will use the zero embedding for OOV
    for embedded_col in embedded_columns.keys():
        features[embedded_col] = df[embedded_col].values

    if not test:
        nlabels = df[TARGET_COL].unique().shape[0]
        # Convert labels to one-hot encodings
        targets = tf.keras.utils.to_categorical(df[TARGET_COL], nlabels)
    else:
        targets = None

    return features, targets



def load_dataset(dataset_dir, batch_size):

    # Read train dataset (and maybe dev, if you need to...)
    dataset, dev_dataset = train_test_split(
        pandas.read_csv(os.path.join(dataset_dir, 'train.csv')), test_size=0.2)
    
    test_dataset = pandas.read_csv(os.path.join(dataset_dir, 'test.csv'))
    
    print('Training samples {}, test_samples {}'.format(
        dataset.shape[0], test_dataset.shape[0]))
    
    return dataset, dev_dataset, test_dataset


def build_model(embedded_columns, direct_features_input, direct_features_input_shape,
               hidden_layer_sizes, dropouts, n_labels):
    tf.keras.backend.clear_session()

    # Add one input and one embedding for each embedded column
    embedding_layers = []
    inputs = []
    for embedded_col, max_value in embedded_columns.items():
        input_layer = layers.Input(shape=(1,), name=embedded_col)
        inputs.append(input_layer)
        # Define the embedding layer
        embedding_size = int(max_value / 4)
        embedding_layers.append(
            tf.squeeze(layers.Embedding(input_dim=max_value, output_dim=embedding_size)(input_layer), axis=-2))
        print('Adding embedding of size {} for layer {}'.format(embedding_size, embedded_col))

    # Add the direct features already calculated
    direct_features_input = layers.Input(shape=direct_features_input_shape, name='direct_features')
    inputs.append(direct_features_input)

    # Concatenate everything together
    features = layers.concatenate(embedding_layers + [direct_features_input])

    if len(hidden_layer_sizes) > 0:
        hidden_layer_size = hidden_layer_sizes.pop(0)
        last_hidden_layer = layers.Dense(hidden_layer_size, activation='relu')(features)
    if len(dropouts) > 0:
        dropout = dropouts.pop(0)
        last_hidden_layer = layers.Dropout(dropout)(last_hidden_layer)
        
    if len(dropouts) > len(hidden_layer_sizes):
        dropouts = dropouts[:len(hidden_layer_sizes)]
        
    for hidden_layer_size, dropout in zip_longest(hidden_layer_sizes, dropouts, fillvalue=None):
        if hidden_layer_size != None:
            last_hidden_layer = layers.Dense(hidden_layer_size, activation='relu')(last_hidden_layer)
        if dropout != None:
            last_hidden_layer = layers.Dropout(dropout)(last_hidden_layer)
        
    output_layer = layers.Dense(n_labels, activation='softmax')(last_hidden_layer)
    return models.Model(inputs=inputs, outputs=output_layer)

def main():
    args = read_args()
    dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, args.batch_size)
    nlabels = dataset[TARGET_COL].unique().shape[0]
    
    # It's important to always use the same one-hot length
    one_hot_columns = {
        one_hot_col: dataset[one_hot_col].max()
        for one_hot_col in args.one_hot_columns
    }
    embedded_columns = {
        embedded_col: dataset[embedded_col].max() + 1
        for embedded_col in args.embedding_columns
    }
    numeric_columns = ['Age', 'Fee']

    # TODO shuffle the train dataset!
    dataset = shuffle(dataset, random_state=47)
    
    # TODO (optional) put these three types of columns in the same dictionary with "column types"
    X_train, y_train = process_features(dataset, one_hot_columns, numeric_columns, embedded_columns)
    direct_features_input_shape = (X_train['direct_features'].shape[1],)
    X_dev, y_dev = process_features(dev_dataset, one_hot_columns, numeric_columns, embedded_columns)
    
    # Create the tensorflow Dataset
    batch_size = args.batch_size

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    dev_ds = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).batch(batch_size)
    X_test, y_test = process_features(
        test_dataset, one_hot_columns, numeric_columns, embedded_columns, test=True)
    test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size)

    # TODO: Build the Keras model
    model = build_model(embedded_columns, X_train['direct_features'], direct_features_input_shape, 
            args.hidden_layer_sizes,
            args.dropout,
            nlabels)

    # TODO: Compile the model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(nested=True):
        # Log model hiperparameters first
        mlflow.log_param('hidden_layer_sizes', args.hidden_layer_sizes)
        mlflow.log_param('dropout', args.dropout)
        mlflow.log_param('embedded_columns', embedded_columns)
        mlflow.log_param('one_hot_columns', one_hot_columns)
        mlflow.log_param('numeric_columns', numeric_columns)
        mlflow.log_param('epochs', args.epochs)

        # Train
        history = model.fit(train_ds, epochs=args.epochs, validation_data=dev_ds, verbose=0)

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

        loss, accuracy = 0, 0
        loss, accuracy = model.evaluate(dev_ds)
        print("*** Dev loss: {} - accuracy: {}".format(loss, accuracy))
        mlflow.log_metric('dev_loss', loss)
        mlflow.log_metric('dev_accuracy', accuracy)
        
        predictions = 'No prediction yet'
        predictions = model.predict(test_ds)

        # TODO: Convert predictions to classes
        # TODO: Save the results for submission
        test_dataset["AdoptionSpeed"] = predictions.argmax(axis=1)
        test_dataset.to_csv("./submission.csv", index=False, columns=["PID", "AdoptionSpeed"])
        print(predictions)

        
print('All operations completed')

if __name__ == '__main__':
    main()
