import numpy as np
#import tensorflow as tf
import os
import io
from gensim.models.fasttext import FastText

def load_ft(filename):
    fname = os.path.abspath(filename)
    model = FastText.load(fname)
    return model

def get_ft(input, ftmodel):
    embed_output = []
    for line in input:
        cur_line = []
        for word in line:
            cur_line.append(ftmodel.wv[str(word)])
        embed_output.append(cur_line)
    return np.asarray(embed_output, dtype=np.float32)


def load_data(filename):
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
                cur_input = []
                len_in = len(line[:-1].split()[:100])
                for i in range(100-len_in):
                    cur_input.append(' ')
                for i in line[:-1].split()[:100]:
                    cur_input.append(i)
                load_input.append(cur_input)
                i = 0

    output_probs = []
    for i in load_output:
        prob = [0 for i in range(len(valid_flairs))]
        prob[i] = 1
        output_probs.append(prob)

    return (np.asarray(load_input, dtype=np.object), np.asarray(output_probs, dtype=np.int32))

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass
