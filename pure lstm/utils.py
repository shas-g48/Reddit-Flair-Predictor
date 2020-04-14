import numpy as np
import tensorflow as tf
import os
import io

def load_vectors(filename):
    """
    Creates vectors for lang in a dict
    of form {word:embedding}
    """

    filein = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')

    data = {}
    for line in filein:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def load_data(filename, en):
    #filein = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    load_input = []
    load_output = []

    valid_flairs = ['Politics', 'Non-Political', 'AskIndia', 'Policy/Economy','Business/Finance','Science/Technology', 'Scheduled', 'Sports', 'Food','Photography','CAA-NRC-NPR', 'Coronavirus']
    map_int = dict()
    for i, j in enumerate(valid_flairs):
        map_int[j] = i

    i = 0
    with open(filename, 'r') as f:
        for line in f:

            if i == 0:
                #print('output: ', line)
                load_output.append(map_int[line[:-1]])
                i = 1

            elif i == 1:
                # truncate sequence to 100 length
                #print('input: ', line)
                load_input.append(line[:-1].split()[:100])
                #print(load_input)
                i = 0

    output_probs = []
    for i in load_output:
        prob = [0 for i in range(len(valid_flairs))]
        prob[i] = 1
        output_probs.append(prob)

    #print('#debug load_input: ', load_input)
    #print('#debug load_output: ', load_output)
    # get input vector

    input_vec = []
    for line in load_input:

        sentence = []
        # left pad input with zeros
        len_input = len(line)

        for i in range(100-len(line)):
            sentence.append(np.zeros(300))

        for word in line:
            sentence.append(en[word])

        input_vec.append(sentence)
        sentence = []
    #print(input_vec)
    print('#debug input vec shape: ', np.asarray(input_vec).shape)

    return (np.asarray(input_vec, dtype=np.float32), np.asarray(output_probs, dtype=np.int32))

def get_data(filename, shuffle, batch_size, en):
    dataset = load_data(filename, en)
    tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    if shuffle == 1:
        tf_dataset = tf_dataset.shuffle(100)
    tf_dataset = tf_dataset.batch(batch_size)

    return tf_dataset

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass
