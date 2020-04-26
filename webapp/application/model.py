import time
import os
import random
import json
from math import ceil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

import application.utils as utils

class FlairPredict(object):
    def __init__(self):
        # learning rate
        self.lr = 1e-3

        #self.batch_size_valid = 500 # big batch size reduces evaluation time

        # batch size for prediction
        self.batch_size_predict = 500

        # timesteps to take in data for
        self.time_steps = 50

        # embedding size used
        self.embed_size = 32

    def get_data(self):
        """
        Load scraped data and reddit urls,
        and make placeholders for input and output
        """
        with tf.variable_scope('data', reuse=tf.AUTO_REUSE) as scope:
            # Load scraped data and their corresponding urls
            self.predict_input, self.predict_url = utils.load_posts('application/reddit_posts/data_query.txt', self.time_steps)

            # placeholders for input and output
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.time_steps, self.embed_size])
            self.output = tf.placeholder(dtype=tf.int32, shape=[None, 12])

    def inference(self):
        """ Define the inference part of the graph """
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE) as scope:


            # gru cell
            cell = tf.contrib.rnn.GRUBlockCellV2(num_units=50)


            # unroll the gru cell
            seq_output, out_state = tf.nn.dynamic_rnn(cell, self.input,
                                                    dtype=tf.float32)

            # get output from end of rnn
            last_output = seq_output[:, self.time_steps - 1, :]   # (bs, hidden_size)

            # 2 fc layers
            int_layer = tf.layers.dense(last_output, 240,
                                            activation=tf.nn.tanh)
            int_layer_2 = tf.layers.dense(int_layer, 120,
                                            activation=tf.nn.tanh)

            # output logits
            self.logits = tf.layers.dense(int_layer_2, 12,
                                            activation=None)

    def evaluate(self):
        """
        Defines operations to find accuracy
        Here, it is just used to get the max indices
        for the logits
        """
        with tf.variable_scope('evaluate', reuse=tf.AUTO_REUSE) as scope:

            # the logits given by the trained model
            self.eval_logits = tf.placeholder(dtype=tf.float32, shape=[None, 12])

            # the correct one hot probability of output
            #self.eval_output = tf.placeholder(dtype=tf.int32, shape=[None, 12])

            # get max value of logit and their index for each input
            max_values, max_indices = tf.nn.top_k(self.eval_logits, k=1)
            self.max_indices = max_indices

    def build(self):
        """ Builds the model """
        self.get_data()
        self.inference()
        self.evaluate()

    def get_batch(self, data, batch_no, batch_size):
        """
        Generates to batch by slicing the data

        data: data to produce batch from
        batch_no: index of batch to produce
        batch_size: size of batch to produce
        """
        return data[batch_no * batch_size : (batch_no + 1) * batch_size]

    def predict(self):
        """
        Runs the pretrained model to give predictions
        in json and python dict format

        format: {'reddit_link':'predicted_flair'}
        """

        # start a tensorflow session
        with tf.Session() as sess:

            # initialize global variables and weight saver
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            # load saved weights if present
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                                    'application/checkpoints/flair/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('loaded saved model!')

            # get number of prediction batches
            total_batches = ceil(len(self.predict_input)/self.batch_size_predict)

            # load fasttext word vector model
            ftmodel = utils.load_ft('application/fasttext.model.bin')

            # hold predictions for all batches
            num_flair = []

            for batch_no in range(total_batches):
                # get input batch
                input_batch = self.get_batch(self.predict_input, batch_no,
                                            self.batch_size_predict)

                # get embeddings for input batch
                input_batch_embeds = utils.get_ft(input_batch, ftmodel)

                # get logits predicted by model
                cur_logits = sess.run(self.logits, feed_dict={self.input:input_batch_embeds})

                # convert the raw logits to max indices
                pred_indices = sess.run(self.max_indices, feed_dict={self.eval_logits:cur_logits})

                # convert column vector to row vector,
                # and remove extra outer dimension
                pred_indices_flat = np.transpose(pred_indices)[0]

                # append predicted flair indices
                # to prediction results
                for index in pred_indices_flat:
                    num_flair.append(index)

            # get mapping from index to flair
            _, index_to_flair = utils.flair_mapping()

            # convert flair indices to flair
            pred_flairs = [index_to_flair[x] for x in num_flair]

            # combine predictions with reddit links
            # with key as link and value as prediction
            pred_object = dict()
            for index, url in enumerate(self.predict_url):
                pred_object[url] = pred_flairs[index]

            print(pred_object)

            # return output both as json and plain dict
            return json.dumps(pred_object), pred_object

def run_model():
    # get the model object and build the computation graph
    model = FlairPredict()
    model.build()

    # get the predictions in json
    # and plain dict and return them
    pred_json, pred_object = model.predict()
    return pred_json, pred_object
