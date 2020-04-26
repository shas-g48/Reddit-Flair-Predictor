import os
import io

import numpy as np
import fasttext


def load_vectors(filename):
    """
    Creates vectors for vocabulary
    in a dict of form {word: embedding}

    filename: file containing vectors
    """

    # load embeddings according to fasttext format
    filein = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    embeds = {}
    for line in filein:
        tokens = line.rstrip().split(' ')
        embeds[tokens[0]] = list(map(float, tokens[1:]))

    # add the embedding for space
    embeds[' '] = [0.016601462,0.0067870123,-0.028963935,0.016884321,-0.008362424,-0.025133897,-0.027598584,-0.01069816,-0.00077911816,-0.023480771,-0.015628096,-0.027175352,0.027422002,0.013819963,0.02896208,-0.02149246,0.015003212,0.006882444,0.01330033,0.0013655132,0.029585376,0.0045530014,0.0024796196,-0.011898544,0.011689002,-0.014970597,-0.00570085,-0.01727686,0.030820029,-0.009950199,-0.0030393065,0.0006726758]
    return embeds

def get_vectors(input_batch, en):
    """
    Convert input batch to embeddings

    input batch: batch consisting of sentences
    en: word vector dictionary
    """

    # hold batch converted to embeddings
    embed_output = []

    # convert one line at a time
    for line in input_batch:
        cur_line = []
        # add embeddings for each word in sentence
        for word in line:
            cur_line.append(en[str(word)])
        # add sentence to batch
        embed_output.append(cur_line)

    return np.asarray(embed_output, dtype=np.float32)

def flair_mapping():
    """ Produces mapping from flair to index and vice versa """
    valid_flairs = ['Politics', 'Non-Political', 'AskIndia', 'Policy/Economy',
                    'Business/Finance','Science/Technology', 'Scheduled',
                     'Sports', 'Food','Photography','CAA-NRC-NPR', 'Coronavirus']

    # create the mappings in separate dictionaries
    flair_to_index = dict()
    index_to_flair = dict()
    for index, flair in enumerate(valid_flairs):
        flair_to_index[flair] = index
        index_to_flair[index] = flair

    return flair_to_index, index_to_flair

def load_data(filename, time_steps):
    """
    Load dataset from file

    filename: file containing dataset
    time_steps: no of words to get for each data point
    """
    # get mapping from flair to index
    flair_to_index, _ = flair_mapping()

    # input and output arrays for data
    load_input = []
    load_output = []

    # a variable to select which array to write to
    select_list = 0
    with open(filename, 'r') as f:
        for line in f:
            # add flair index to output and toggle
            if select_list == 0:
                load_output.append(flair_to_index[line[:-1]])
                select_list = 1 # the next line is input

            # add sentence to input and toggle
            elif select_list == 1:
                # hold current input
                cur_input = []

                # get number of words in current line
                len_in = len(line[:-1].split()[:time_steps])

                # add spaces before the sentence to get total length to timesteps
                for _ in range(time_steps - len_in):
                    cur_input.append(' ')

                # add the words to complete the input line
                for word in line[:-1].split()[:time_steps]:
                    cur_input.append(word)

                load_input.append(cur_input)

                select_list = 0 # the next line is output

    # hold outputs in one hot format
    one_hot_output = []
    # convert output flair indices to one hot, one output at a time
    for output_flair_index in load_output:
        # generate zeros of size no of flairs, and one hot it according to index
        one_hot = [0 for i in range(12)]
        one_hot[output_flair_index] = 1

        # append to output array
        one_hot_output.append(one_hot)

    return (np.asarray(load_input, dtype=np.object), np.asarray(one_hot_output, dtype=np.int32))

def safe_mkdir(path):
    """
    Creates directories if they don't exist

    path: relative path to directory
    """
    try:
        os.mkdir(path)
    except OSError:
        pass

## Note: below utilities are for webapp
def load_ft(filename):
    """
    Loads the fasttext wordvector model

    filename: name of model
    """
    fname = os.path.abspath(filename)
    model = fasttext.load_model(fname)
    return model

def get_ft(input, ftmodel):
    """
    Get embeddings for input from fasttext model

    input: input to get embeddings for
    ftmodel: a fasttext model object
    """

    # hold batch converted to embeddings
    embed_output = []

    # convert one line at a time
    for line in input:
        cur_line = []
        # add embeddings for each word in sentence
        for word in line:
            cur_line.append(ftmodel.get_word_vector(str(word)))
        # add sentence to batch
        embed_output.append(cur_line)

    return np.asarray(embed_output, dtype=np.float32)
