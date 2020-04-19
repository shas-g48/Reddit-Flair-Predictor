import numpy as np
#import tensorflow as tf
import os
import io
from gensim.models.fasttext import FastText

import requests
import json
import time
import datetime
import praw
import nltk
import string

def process_text(text):
    import re
    # remove urls from text
    text = re.sub(r'https\S+', '', text)
    tokens = nltk.tokenize.word_tokenize(text)
    # remove usernames of form u/username from text
    tokens = [re.sub(r'u\/.*', '', i) for i in tokens]
    # remove subreddit of form r/subreddit from text
    tokens = [re.sub(r'r\/.*', '', i) for i in tokens]

    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    #words = [word for word in stripped if word.isalpha()]
    words = [re.sub(r'[0-9]', '', i) for i in stripped]

    #text = ' '.join(words)
    words = [x for x in words if x!= '']
    text = ' '.join(words)
    return text

def keep_alpha(text):
    tokens = nltk.tokenize.word_tokenize(text)
    words = [x for x in tokens if x.isalpha()]
    text = ' '.join(words)
    return text

def get_reddit_data(query_urls):
    reddit = praw.Reddit()

    posts = []
    for url in query_urls:
        submission = reddit.submission(url=url)

        d = dict()

        d['title'] = process_text(submission.title)
        d['flair'] = submission.link_flair_text
        d['selftext'] = process_text(submission.selftext)
        d['date'] = datetime.datetime.fromtimestamp(submission.created)
        d['sub_id'] = submission.id
        d['date_raw'] = submission.created
        d['url'] = url

        posts.append(d)

    return posts

def write_reddit_data(data):
    with open('application/reddit_posts/data_query.txt', 'w') as f:
        for post in data:
            f.write(str(post['url']))
            f.write('\n')
            f.write(str(keep_alpha(post['title'])))
            f.write('\n')

def load_posts(filename):
    load_input = []
    load_url = []

    i = 0
    with open(filename, 'r') as f:
        for line in f:
            if i == 0:
                load_url.append(line[:-1])
                i = 1
            elif i == 1:
                cur_input = []
                len_in = len(line[:-1].split()[:25])
                for i in range(25-len_in):
                    cur_input.append(' ')
                for i in line[:-1].split()[:25]:
                    cur_input.append(i)
                load_input.append(cur_input)
                i = 0

    return (np.asarray(load_input, dtype=np.object), np.asarray(load_url, dtype=np.object))


def load_ft(filename):
    fname = os.path.abspath(filename)
    model = FastText.load(fname)
    return model

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
    data[' '] = [0.0069633694,-0.0033999255,0.010586381,-0.0006393717,0.011965754,-0.0009611225,-0.017290328,-0.0025544537,-0.00087693986,-0.0020554701,-0.01321163,0.00012053901,0.013346921,-0.006311356,0.003211852,-0.015739692,-0.008041525,-0.01571059,0.0057985648,-0.013242714,-0.0023971163,-0.0074268393,0.004600727,0.0025885534,0.012782362,-0.0062122196,0.002212011,-0.016663603,0.0003671778,0.017689114,0.016025512,-0.01483213,0.015711218,-0.01036233,0.0066601257,-0.009123624,0.0037135174,-0.0069212536,0.010281087,-0.0035295007,0.002849685,-0.010490801,0.0026974797,-0.013000383,0.0036737898,0.015114544,0.017116468,-0.006816537,-0.0074365637,-0.008343396]
    return data

def get_vectors(input, en):
    embed_output = []
    for line in input:
        cur_line = []
        for word in line:
            #print('#debug word: ', str(word))
            cur_line.append(en[str(word)])
        embed_output.append(cur_line)

    return np.asarray(embed_output, dtype=np.float32)

def get_ft(input, ftmodel):
    embed_output = []
    for line in input:
        cur_line = []
        for word in line:
            cur_line.append(ftmodel.wv[str(word)])
        embed_output.append(cur_line)
    return np.asarray(embed_output, dtype=np.float32)

def flair_mapping():
    valid_flairs = ['Politics', 'Non-Political', 'AskIndia', 'Policy/Economy','Business/Finance','Science/Technology', 'Scheduled', 'Sports', 'Food','Photography','CAA-NRC-NPR', 'Coronavirus']
    map_int = dict()
    int_map = dict()
    for i, j in enumerate(valid_flairs):
        map_int[j] = i
        int_map[i] = j
    return map_int, int_map

def load_data(filename):
    load_input = []
    load_output = []

    map_int, _ = flair_mapping()


    print(map_int)
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
                len_in = len(line[:-1].split()[:25])
                for i in range(25-len_in):
                    cur_input.append(' ')
                for i in line[:-1].split()[:25]:
                    cur_input.append(i)
                load_input.append(cur_input)
                i = 0

    output_probs = []
    #print('#debug load_output: ', load_output)
    for i in load_output:
        prob = [0 for i in range(len(valid_flairs))]
        prob[i] = 1
        output_probs.append(prob)
    #print('#debug load_output_probs: ', output_probs)

    return (np.asarray(load_input, dtype=np.object), np.asarray(output_probs, dtype=np.int32))

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass
