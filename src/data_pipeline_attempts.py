import sys
sys.path.append(r"C:\Users\santo\workspace\machine-translation-using-transformers")

import tensorflow as tf
import collections
from utils import dataset as data
import numpy as np
import yaml

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import TextVectorization

from utils.embedding import Glove

tf.py_function
def line_process(line1, line2):
    # input_words = list(line1)[0].split()
    # label_words = list(line2)[0].split()
    
    # glove = Glove(parameters)
    # vectors_input = [glove.embed(word) for word in input_words]
    # vectors_output = [glove.embed(word) for word in label_words]
    # return (vectors_input, vectors_output)
    return line1, line2

def load_as_tf_data(parameters):
    # Load English data
    inputs = tf.data.TextLineDataset(parameters['dataset']['data_dir']['input'])

    # load output language data
    outputs = tf.data.TextLineDataset(parameters['dataset']['data_dir']['output'])

    # tf.data preprocessing
    dataset = tf.data.Dataset.zip((inputs, outputs))
    dataset = dataset.map(line_process)
    dataset = dataset.prefetch(100)
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(parameters['dataset']['batch_size'])

    train_size = int(0.8 * parameters['dataset']['data_size'])
    val_size = int(0.2 * parameters['dataset']['data_size'])
    
    train_ds = dataset.take(train_size)    
    val_ds = dataset.skip(train_size).take(val_size)

    return dataset, train_ds, val_ds

if __name__=="__main__":
    print(tf.test.is_gpu_available())
    with open('../parameters.yaml', 'r') as file:
        parameters = yaml.safe_load(file)

    dataset, train_ds, val_ds = load_as_tf_data(parameters)

    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)

    

    print('Dataset Loaded', dataset, train_ds, val_ds, "Samples")
