import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import numpy as np
import tensorflow as tf
import random
from math import ceil
import time

import utils

class FlairPredict(object):

    def __init__(self):
        self.lr = 3e-4
        self.gstep = tf.get_variable('gstep', dtype=tf.int32, initializer=tf.constant(0))
        #self.en = utils.load_vectors('data/vocab.vec')
        self.batch_size = 16    # batch size for training
        self.batch_size_test = 100
        self.skip_step = 500    # to print loss at regular intervals
        self.size = 86788    # training dataset size
        self.size_test = 235
        #self.en_vec = tf.get_variable('en_vec', dtype=tf.float32,initializer=tf.constant(list(self.en.values())))
        self.init_state = tf.get_variable('init_state', dtype=tf.float32, shape = (self.batch_size, 1000), initializer=tf.zeros_initializer())

    def get_data(self):
        with tf.name_scope('data'):
            self.train_input, self.train_output = utils.load_data('data/data2_cleaned.txt')
            #print(self.train_input)
            #print(self.train_output)
            #self.input = tf.placeholder(dtype=tf.float32, shape=[None, 100, 300])
            #self.input = tf.placeholder(dtype=tf.float32, shape=[None, 50, 300])
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, 50, 100])
            self.output = tf.placeholder(dtype=tf.int32, shape=[None, 12])


    def inference(self):
        with tf.name_scope('inference'):
            #cell = tf.nn.rnn_cell.GRUCell(200)
            hidden_sizes = [75, 75, 75]
            layers = [tf.nn.rnn_cell.GRUCell(size) for size in hidden_sizes]
            cell = tf.nn.rnn_cell.MultiRNNCell(layers)
            seq_output, out_state = tf.nn.dynamic_rnn(cell, self.input, dtype=tf.float32)#, initial_state=self.init_state)
            self.last_output = seq_output[:,49,:]
            #self.last_output = seq_output[:,99,:]
            self.logits = tf.layers.dense(self.last_output, 12, activation=None)

    def loss(self):
        with tf.name_scope('loss'):
            probs = tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=self.logits)
            self.loss = tf.reduce_mean(probs)

    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

    def build(self):
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()

    def train(self, num_epochs):
        with tf.Session() as sess:

            # Shuffle the dataset
            data = []
            for i in range(len(self.train_input)):
                elem = (self.train_input[i], self.train_output[i])
                data.append(elem)
                #print(data)
            #print(data)
            random.shuffle(data)


            data_input = []
            data_output = []

            for i in range(len(data)):
                data_input.append(data[i][0])
                data_output.append(data[i][1])

            del data

            # load fastText model
            ftmodel = utils.load_ft('fasttext.model')

            sess.run(tf.global_variables_initializer())

            for j in range(num_epochs):
                total_batches = ceil(len(self.train_input)/self.batch_size)
                print('#debug epoch: ', j)
                total_loss = 0
                start_time = time.time()
                for i in range(total_batches):

                        # batch the data
                        #start_time = time.time()
                        bs_input = data_input[i*self.batch_size:(i+1)*self.batch_size]
                        bs_output = np.asarray(data_output[i*self.batch_size:(i+1)*self.batch_size], dtype=np.int32)

                        # convert input batch into fasttext embeddings
                        bs_input_ft = utils.get_ft(bs_input, ftmodel)
                        #end_time = time.time()
                        #print('#debug embed: ', end_time-start_time)

                        input, logits, _, loss = sess.run([self.input, self.logits, self.opt, self.loss], feed_dict={self.input:bs_input_ft, self.output:bs_output})
                        #loss_time = time.time()
                        #print('#debug loss time: ', loss_time-end_time)

                        total_loss += loss
                        #print(np.asarray(input).shape)
                        #print('#debug input: ', input)
                        #print(logits.shape)
                        #print('#debug output: ', logits)
                        if i % 10 == 0:
                            print('#debug loss: ', loss)
                print('time to train one epoch {} : {}'.format(j, time.time()-start_time))
                print("average loss at epoch {} : {}".format(j, total_loss/total_batches))


if __name__ == '__main__':
    # set random seeds
    np.random.seed(seed)
    tf.set_random_seed(seed)

    model = FlairPredict()
    model.build()
    model.train(100)
    print('#debug: model built!')
