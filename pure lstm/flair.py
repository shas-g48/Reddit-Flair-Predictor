import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import numpy as np
import tensorflow as tf

import utils

class FlairPredict(object):

    def __init__(self):
        self.lr = 1e-3
        self.gstep = tf.get_variable('gstep', dtype=tf.int32, initializer=tf.constant(0))
        self.en = utils.load_vectors('data/vocab.vec')
        self.batch_size = 16    # batch size for training
        self.batch_size_test = 100
        self.skip_step = 500    # to print loss at regular intervals
        self.size = 86788    # training dataset size
        self.size_test = 235
        #self.en_vec = tf.get_variable('en_vec', dtype=tf.float32,initializer=tf.constant(list(self.en.values())))
        self.init_state = tf.get_variable('init_state', dtype=tf.float32, shape = (self.batch_size, 1000), initializer=tf.zeros_initializer())

    def get_data(self):
        with tf.name_scope('data'):
            train_data = utils.get_data('data/data2_cleaned.txt', 1, self.batch_size, self.en)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

            self.input, self.output = iterator.get_next()
            self.input_shape = self.input.get_shape()
            print(self.input_shape)

            self.train_init = iterator.make_initializer(train_data)

    def inference(self):
        with tf.name_scope('inference'):
            cell = tf.nn.rnn_cell.GRUCell(1000)
            self.output, out_state = tf.nn.dynamic_rnn(cell, self.input, (100,16), self.init_state)

    def build(self):
        self.get_data()
        self.inference()

if __name__ == '__main__':
    model = FlairPredict()
    model.build()
    print('#debug: model built!')
