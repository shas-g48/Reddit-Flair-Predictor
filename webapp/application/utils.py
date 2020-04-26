import os
import io
import requests
import json
import time
import datetime
import string
import re

import praw
import nltk
import numpy as np
import fasttext

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

def process_text(text):
    """
    Process data by removing undesirable characters
    text: string to be processed
    """
    # remove urls from text and tokenize it
    text = re.sub(r'https\S+', '', text)
    tokens = nltk.tokenize.word_tokenize(text)

    # remove reddit username and subreddit mentions
    tokens = [re.sub(r'u\/.*', '', i) for i in tokens]
    tokens = [re.sub(r'r\/.*', '', i) for i in tokens]

    # remove punctuations
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove numeric characters from words
    words = [re.sub(r'[0-9]', '', i) for i in stripped]

    # remove empty tokens and join text
    words = [x for x in words if x!= '']
    text = ' '.join(words)

    return text

def keep_alpha(text):
    """
    Keep only alphabets

    text: string to process
    this function keeps compatibility with the
    data processing done in eda after scraping
    """
    # tokenize the text and keep only alphabets
    tokens = nltk.tokenize.word_tokenize(text)
    words = [x for x in tokens if x.isalpha()]

    # separate the words with spaces and return it
    text = ' '.join(words)
    return text

def get_reddit_data(query_urls):
    """
    Scrapes reddit to get desired posts

    query_urls: list of reddit links
    """
    # get a reddit instance
    reddit = praw.Reddit(client_id='WI6HW6PU0KWLtQ',
                client_secret='URXonv9_7VKTgARm3V-tAXH_Jco',
                 user_agent='indiasubscrape')

    posts = [] # hold post dicts

    # go over urls, one at a time
    for url in query_urls:
        # get reddit submission for the url
        submission = reddit.submission(url=url)

        # hold data for current submission
        d = dict()

        # map fetched attributes to desired attributes
        d['title'] = process_text(submission.title)
        d['flair'] = submission.link_flair_text
        d['selftext'] = process_text(submission.selftext)
        d['date'] = datetime.datetime.fromtimestamp(submission.created)
        d['sub_id'] = submission.id
        d['date_raw'] = submission.created
        d['url'] = url

        # add the data to posts
        posts.append(d)

    return posts

def write_reddit_data(data):
    """
    Writes scraped data to disk for
    the prediction model to read

    data: a list of dicts, each dict
          contains data scraped for url
    """
    # writeout scraped data to disk
    with open('application/reddit_posts/data_query.txt', 'w') as f:
        for post in data:
            f.write(str(post['url']))
            f.write('\n')
            f.write(str(keep_alpha(post['title'])))
            f.write('\n')

def load_posts(filename, time_steps):
    """
    Load scraped data from disk
    and return title and url for each
    post

    filename: file to load dataset
    time_steps: no of words to get for each data point
    """

    # hold titles and urls for each post
    load_input = []
    load_url = []

    # a variable to select which array to write to
    select_list = 0
    with open(filename, 'r') as f:
        for line in f:
            # add to url list and toggle
            if select_list == 0:
                load_url.append(line[:-1])
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
                select_list = 0 # the next line is url

    return (np.asarray(load_input, dtype=np.object), np.asarray(load_url, dtype=np.object))

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
    ftmodel: a fasttext model
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
