import time
import random
from math import ceil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

import utils_new as utils


class FlairPredict(object):
    def __init__(self):
        # learning rate
        self.lr = 1e-3

        # step count for summaries used in tensorboard
        self.gstep = tf.get_variable('gstep', dtype=tf.int32,
                                    initializer=tf.constant(0))
        self.gstep_acc = tf.get_variable('gstep_acc', dtype=tf.int32,
                                    initializer=tf.constant(0))

        # get vectors for vocabulary used while training
        self.en = utils.load_vectors('data/vocab.vec')

        # batch size for training and validation
        self.batch_size_train = 32
        self.batch_size_valid = 10000

        # to print loss at regular intervals
        self.skip_step = 50

        # timesteps to take in data for
        self.time_steps = 50

        # embedding size used
        self.embed_size = 32

        # count number of times above
        # desired validation accuracy
        self.above_acc = 0

        # hold maximum validation accuracy
        # and no of times it was lower than
        # the latest max validation accuracy
        self.max_acc = 0
        self.overfit = 0

    def get_data(self):
        """ Load the dataset, and make placeholders for input and output """
        with tf.variable_scope('data', reuse=tf.AUTO_REUSE) as scope:
            # Load datasets for training and validation
            self.train_input, self.train_output = utils.load_data('data/data7_train_more.txt',
                                                                    self.time_steps)
            self.valid_input, self.valid_output = utils.load_data('data/data7_valid.txt',
                                                                    self.time_steps)

            # get validation set size, used for printing progress
            self.size_valid = len(self.valid_input)

            # placeholders for input and output
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.time_steps, self.embed_size])
            self.output = tf.placeholder(dtype=tf.int32, shape=[None, 12])

    def inference(self):
        """ Define the inference part of the graph """
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE) as scope:

            # placeholders for dropout rates
            self.last_output_rate = tf.placeholder(dtype=tf.float32, shape=())
            self.dense1_rate = tf.placeholder(dtype=tf.float32, shape=())
            self.dense2_rate = tf.placeholder(dtype=tf.float32, shape=())

            # gru cell
            cell = tf.keras.layers.GRUCell(
                        units=50, kernel_regularizer=tf.keras.regularizers.l2(0.6),
                        recurrent_regularizer=tf.keras.regularizers.l2(0.6)
                        )

            # attn = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=7)

            # unroll the gru cell and get rnn outputs
            rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
            seq_output, out_state = rnn(self.input)

            # get output from the end of rnn
            last_output = seq_output[:, self.time_steps - 1, :]   # (bs, hidden_size)

            # dropout for output from end of rnn
            last_output_drop = tf.nn.dropout(last_output, rate=self.last_output_rate)

            # dense layer 1 with dropout
            dense1 = tf.layers.dense(last_output_drop, 120, activation=tf.nn.tanh,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.6))
            dense1_drop = tf.nn.dropout(dense1, rate=self.dense1_rate)

            # dense layer 2 with dropout
            dense2 = tf.layers.dense(dense1_drop, 240, activation=tf.nn.tanh,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.6))
            dense2_drop = tf.nn.dropout(dense2, rate=self.dense2_rate)

            # output logits
            self.logits = tf.layers.dense(dense2_drop, 12, activation=None,
                                kernel_regularizer=tf.keras.regularizers.l2(l=0.6))


    def loss(self):
        """
        Defines loss used to train
        softmax cross entropy with logits is used to compute loss
        between network output and the correct one hot output
        """
        with tf.name_scope('loss'):
            # get the loss for each input
            log_prob = tf.nn.softmax_cross_entropy_with_logits(labels=self.output,
                                                                logits=self.logits) # (bs, 12)

            # take mean across inputs
            self.loss = tf.reduce_mean(log_prob)

    def optimize(self):
        """ Defines optimizer operation """
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                            global_step=self.gstep)

    def evaluate(self):
        """
        Defines operations to find accuracy

        self.acc: No of correct results in whole validation set
        self.cat_acc: No of correct results in whole validation set by flair
        """
        with tf.variable_scope('evaluate', reuse=tf.AUTO_REUSE) as scope:

            # hold total accuracy and accuracy by category (flair)
            self.acc = tf.get_variable('acc', initializer=tf.constant(0))
            self.cat_acc = tf.get_variable('cat_acc', shape=(1,12),
                                            dtype=tf.int32,
                                            initializer=tf.zeros_initializer())

            # construct default one hot for all indices to gather from
            # as tf.one_hot does not allow specifying depth at runtime
            default_indices = tf.get_variable('default_indices',
                                            initializer=tf.constant([0,1,2,3,4,
                                            5,6,7,8,9,10,11]), trainable=False)
            value_on = tf.get_variable('value_on', dtype=tf.int32,
                                            initializer=tf.constant(1),
                                            trainable=False)
            value_off = tf.get_variable('value_off', dtype=tf.int32,
                                            initializer=tf.constant(0),
                                            trainable=False)
            self.one_hot_default = tf.one_hot(default_indices, 12,
                                            on_value=value_on,
                                            off_value=value_off)

            # the logits given by the trained model
            self.eval_logits = tf.placeholder(dtype=tf.float32, shape=[None, 12])

            # the correct one hot probability of output
            self.eval_output = tf.placeholder(dtype=tf.int32, shape=[None, 12])

            # get maximum value of logit and their index for each input
            max_values, max_indices = tf.nn.top_k(self.eval_logits, k=1)
            self.max_indices = max_indices

            # convert indices for maximum logit to one hot encoding
            # this operation gathers from one hot default
            one_hot = tf.gather_nd(self.one_hot_default, indices=max_indices,
                                    name='one_hot')
            # self.one_hot = one_hot

            # elementwise comparison of the one hot vectors
            one_hot_eq = tf.equal(one_hot, self.eval_output)
            # self.one_hot_eq = one_hot_eq

            # convert to a single truth value for which input
            eq_row = tf.reduce_all(one_hot_eq, axis=1)
            self.eq_row = eq_row

            # convert truth values to scalar and sum to get no of correct preds
            correct = tf.reduce_sum(tf.where(eq_row, tf.fill(tf.shape(eq_row), 1),
                                                    tf.fill(tf.shape(eq_row), 0)))
            # self.correct = correct

            # update total accuracy by adding accuracy of current batch
            self.acc_upd = self.acc.assign_add(correct)

            # get one hot vectors corresponding to correct predictions by model
            indices_correct = tf.where(eq_row)
            # self.indices_correct = indices_correct
            # self.one_hot_correct = one_hot_correct

            one_hot_correct = tf.gather(self.eval_output, indices_correct)
            # add the gathered vectors to get no of correct predictions per
            # category (flair) in current batch
            correct_cat = tf.reduce_sum(one_hot_correct, axis=0)
            # self.correct_cat = correct_cat

            # update accuracy by category by adding accuracy of current batch
            self.cat_acc_upd = self.cat_acc.assign_add(correct_cat)

            # operations to reset both accuracies
            self.acc_reset = tf.assign(self.acc, tf.constant(0))
            self.cat_acc_reset = tf.assign(self.cat_acc, tf.zeros(shape=(1,12),
                                            dtype=tf.int32))

    def get_batch(self, data, batch_no, batch_size):
        """
        Generates to batch by slicing the data

        data: data to produce batch from
        batch_no: index of batch to produce
        batch_size: size of batch to produce
        """
        return data[batch_no * batch_size : (batch_no + 1) * batch_size]

    def eval_once(self, sess, write_preds):
        """
        Runs the operations defined in evaluate() to find accuracies

        sess: tensorflow session
        write_preds: whether to write prediction outcome to file
        """

        # reset accuracies
        sess.run([self.acc_reset, self.cat_acc_reset])

        # find number of batches
        total_batches = ceil(len(self.valid_input) / self.batch_size_valid)

        # save start time
        start_time_eval = time.time()

        # hold prediction outcome for complete validation set
        bool_result = []

        # compute accuracy over batches
        for batch_no in range(total_batches):
            # get input and output batch
            input_batch = self.get_batch(self.valid_input, batch_no,
                                        self.batch_size_valid)
            output_batch = np.asarray(self.get_batch(self.valid_output,
                                        batch_no, self.batch_size_valid))

            # get embeddings for input batch
            input_batch_embeds = utils.get_vectors(input_batch, self.en)

            # get logits predicted by model
            cur_logits = sess.run(self.logits,
                                feed_dict={self.input:input_batch_embeds,
                                self.dense1_rate:0, self.dense2_rate:0,
                                 self.last_output_rate:0})

            # get boolean prediction result, and updated accuracies
            pred_result, total_acc, cat_acc =  sess.run([self.eq_row, self.acc_upd,
                                                        self.cat_acc_upd],
                                                        feed_dict={self.eval_logits:cur_logits,
                                                        self.eval_output:output_batch})

            # save boolean prediction results if want to save pred outcomess
            if write_preds == True:
                for pred in pred_result:
                    bool_result.append(pred)

        # write out prediction outcomes to file
        if write_preds == True:
            with open('results_valid.txt', 'w') as f:
                for pred in bool_result:
                    f.write(str(pred))
                    f.write('\n')

        # get accuracy
        validation_acc = total_acc / self.size_valid

        # print time to evaluate
        print('#debug time to eval: ', time.time()-start_time_eval)

        # print total, by flair and percentage accuracies
        print('#debug got total correct {} out of {}: '.format(total_acc,
                                                            self.size_valid))
        print('#debug correct by category: ', cat_acc)
        print("#debug % accuracy: {}".format(total_acc / self.size_valid))

        # terminate training if achieve desired
        # validation accuracy 3 times
        if validation_acc > 0.93:
            self.above_acc += 1
        if self.above_acc == 3:
            exit()

        # if validation acc exceeds max current
        # training going okay and reset overfit counter
        if validation_acc > self.max_acc:
            self.max_acc = validation_acc
            self.overfit = 0

        # else increase overfit counter
        else:
            self.overfit += 1

        # terminate training if validation acc does
        # not exceed last max for 8 epochs
        if self.overfit == 15:
            print('Overfit!')
            exit()

    def summary(self):
        """
        Defines summary operations for tensorboard
        Summaries for loss per update,
        total and by flair accuracy per epoch
        average loss per epoch
        """
        with tf.name_scope('summaries'):
                # summary for loss per weight update
                self.loss_summary = tf.summary.scalar('loss', self.loss)

                # summaries for total accuracy and accuracy by category per epoch
                total_acc_summary = tf.summary.scalar('total_acc', self.acc)
                cat0_summary = tf.summary.scalar('cat0_acc', self.cat_acc[0][0])
                cat1_summary = tf.summary.scalar('cat1_acc', self.cat_acc[0][1])
                cat2_summary = tf.summary.scalar('cat2_acc', self.cat_acc[0][2])
                cat3_summary = tf.summary.scalar('cat3_acc', self.cat_acc[0][3])
                cat4_summary = tf.summary.scalar('cat4_acc', self.cat_acc[0][4])
                cat5_summary = tf.summary.scalar('cat5_acc', self.cat_acc[0][5])
                cat6_summary = tf.summary.scalar('cat6_acc', self.cat_acc[0][6])
                cat7_summary = tf.summary.scalar('cat7_acc', self.cat_acc[0][7])
                cat8_summary = tf.summary.scalar('cat8_acc', self.cat_acc[0][8])
                cat9_summary = tf.summary.scalar('cat9_acc', self.cat_acc[0][9])
                cat10_summary = tf.summary.scalar('cat10_acc', self.cat_acc[0][10])
                cat11_summary = tf.summary.scalar('cat11_acc', self.cat_acc[0][11])
                self.summary_acc = tf.summary.merge([cat0_summary, cat1_summary,
                                                    cat2_summary, cat3_summary,
                                                    cat4_summary, cat5_summary,
                                                    cat6_summary, cat7_summary,
                                                    cat8_summary, cat9_summary,
                                                    cat10_summary, cat11_summary,
                                                    total_acc_summary])

                # summary for average loss per epoch
                self.loss_avg_summary = tf.summary.scalar('loss_avg', self.loss_avg)

    def loss_avg(self):
        """ Defines operations to compute average loss per epoch """
        # placeholders for total loss and number of batches
        self.total_loss = tf.placeholder(dtype=tf.float32, shape=())
        self.total_batches = tf.placeholder(dtype=tf.float32, shape=())

        # compute average loss
        self.loss_avg = self.total_loss / self.total_batches

    def build(self):
        """ Builds the model """
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.evaluate()
        self.loss_avg()
        self.summary()

    def train(self, num_epochs):
        """
        Trains the model

        num_epochs: number of epochs to train
        """

        # make directories to save weights and summaries
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/flair')
        writer = tf.summary.FileWriter('./graphs/flair', tf.get_default_graph())

        # start a tensorflow session
        with tf.Session() as sess:

            # initialize global variables and get weight saver
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            # load saved weights if present
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                                                'checkpoints/flair/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('#debug: loaded saved model!')

            # get step count for tensorboard summaries
            step = self.gstep.eval()
            step_acc = self.gstep_acc.eval()

            # Convert dataset to input output tuples to facilitate shuffling
            data = []
            for index in range(len(self.train_input)):
                data.append((self.train_input[index], self.train_output[index]))


            # evaluate before training
            self.eval_once(sess, False)

            # uncomment to test
            #exit()

            # iterate over epochs
            for epoch in range(num_epochs):
                # change mode to training
                #self.change_mode(saver, sess, 'train')

                # get number of training batches and print it
                total_batches = ceil(len(self.train_input) / self.batch_size_train)
                print('#debug epoch: {} no of batches: {}'.format(epoch, total_batches))

                # hold total loss for epoch
                total_loss = 0

                # shuffle the dataset
                random.shuffle(data)

                # get separate input and output arrays from tuples
                data_input = []
                data_output = []
                for index in range(len(data)):
                    data_input.append(data[index][0])
                    data_output.append(data[index][1])

                # save start time
                start_time = time.time()

                # iterate over batches
                for batch_no in range(total_batches):

                    # get input and output batch
                    input_batch = self.get_batch(data_input, batch_no,
                                                self.batch_size_train)
                    output_batch = np.asarray(self.get_batch(data_output, batch_no,
                                            self.batch_size_train), dtype=np.int32)

                    # get embeddings for input batch
                    input_batch_embeds = utils.get_vectors(input_batch, self.en)

                    # run optimizer, get loss and loss summary
                    _, loss, loss_summary = sess.run([self.opt, self.loss,
                                                    self.loss_summary],
                                                    feed_dict={self.input:input_batch_embeds,
                                                    self.output:output_batch,
                                                    self.dense1_rate:0,
                                                    self.dense2_rate:0,
                                                    self.last_output_rate:0})

                    # add loss summary per update
                    writer.add_summary(loss_summary, global_step=step)

                    # update total loss and step for total loss
                    total_loss += loss
                    step += 1

                    # print loss at regular intervals
                    if batch_no % self.skip_step == 0:
                        print('#debug loss: ', loss)

                # print time to train epoch and average loss
                print('time to train one epoch {} : {}'.format(epoch, time.time()
                                                                - start_time))
                print("average loss at epoch {} : {}".format(epoch, total_loss
                                                                / total_batches))

                # save weights at every epoch
                saver.save(sess, 'checkpoints/flair/checkpoint', step)

                # get validation accuracy at every epoch
                self.eval_once(sess, write_preds=False)

                # get summaries for accuracies and average loss per epoch
                summary_acc = sess.run(self.summary_acc)
                summary_loss_avg = sess.run(self.loss_avg_summary,
                                            feed_dict={self.total_loss:total_loss,
                                             self.total_batches:total_batches})

                # add average loss and accuracies summaries
                writer.add_summary(summary_loss_avg, global_step=step_acc)
                writer.add_summary(summary_acc, global_step=step_acc)

                # update step for summaries per epoch
                step_acc += 1

            # uncomment to write validation preds at end of training
            #self.eval_once(sess, write_preds=False)

if __name__ == '__main__':
    # set random seeds
    #seed = 10
    #seed = 83
    #seed = 152
    #np.random.seed(seed)
    #tf.set_random_seed(seed)
    #random.seed(seed)

    # get the model object and build the computation graph
    model = FlairPredict()
    model.build()
    print('#debug: model built!')

    # train the model
    model.train(810)
    print('#debug: model trained!')
