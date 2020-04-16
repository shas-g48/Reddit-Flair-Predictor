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
        self.gstep_acc = tf.get_variable('gstep_acc', dtype=tf.int32, initializer=tf.constant(0))
        #self.en = utils.load_vectors('data/vocab.vec')
        self.batch_size = 16    # batch size for training
        self.batch_size_valid = 500 # big batch size reduces evaluation time
        self.skip_step = 500    # to print loss at regular intervals
        self.size = 69422    # training dataset size
        self.size_valid = 17366
        #self.en_vec = tf.get_variable('en_vec', dtype=tf.float32,initializer=tf.constant(list(self.en.values())))
        self.init_state = tf.get_variable('init_state', dtype=tf.float32, shape = (self.batch_size, 1000), initializer=tf.zeros_initializer())

    def get_data(self):
        with tf.name_scope('data'):
            self.train_input, self.train_output = utils.load_data('data/data_correct_med.txt')
            self.valid_input, self.valid_output = utils.load_data('data/data_valid.txt')
            #print(self.train_input)
            #print(self.train_output)
            #self.input = tf.placeholder(dtype=tf.float32, shape=[None, 100, 300])
            #self.input = tf.placeholder(dtype=tf.float32, shape=[None, 50, 300])
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, 25, 50])
            self.output = tf.placeholder(dtype=tf.int32, shape=[None, 12])

    def inference(self):
        with tf.name_scope('inference'):
            cell = tf.nn.rnn_cell.GRUCell(30)
            #hidden_sizes = [75, 75, 75]
            #layers = [tf.nn.rnn_cell.GRUCell(size) for size in hidden_sizes]
            #cell = tf.nn.rnn_cell.MultiRNNCell(layers)
            # seq_output is of dims (bs, seq_len, hidden_units)
            seq_output, out_state = tf.nn.dynamic_rnn(cell, self.input, dtype=tf.float32)#, initial_state=self.init_state)
            self.last_output = seq_output[:,24,:]

            self.seq_output = seq_output
            #self.last_output = seq_output[:,99,:]
            # output dims should be (bs, 12), input has dims (bs, hidden_units)
            #self.logits = tf.layers.dense(self.last_output, 12, activation=None)
            #bias_logits = [1.2658978942202606, 1.3579852257480152, 1.7962401566791706, 2.8882079026286114, 2.361042985190914, 2.9551776328319828, 5.027343208116932, 3.7500284794334178, 4.343022210185258, 3.7096965608847454, 7.815875580753849, 3.693360141565052]
            self.logits = tf.layers.dense(self.last_output, 12, activation=None)


    def loss(self):
        with tf.name_scope('loss'):
            probs = tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=self.logits)
            # should have dims (bs, 12)
            self.probs = probs
            self.loss = tf.reduce_mean(probs)

    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

    def evaluate(self):
        with tf.name_scope('evaluate'):
            # to hold total accuracy of the whole validation set
            self.acc = tf.get_variable('acc', initializer=tf.constant(0))
            # to hold accuracy of whole validation set by category
            self.cat_acc = tf.get_variable('cat_acc', shape=(1,12), dtype=tf.int32, initializer=tf.zeros_initializer())

            # construct default one hot to gather from as
            # tf.one_hot does not allow specifying depth at runtime
            default_indices = tf.get_variable('default_indices',
                            initializer=tf.constant([0,1,2,3,4,5,6,7,8,9,10,11]))
            # specify on value and off value of int32 type
            value_on = tf.get_variable('value_on', dtype=tf.int32, initializer=tf.constant(1))
            value_off = tf.get_variable('value_off', dtype=tf.int32, initializer=tf.constant(0))
            self.one_hot_default = tf.one_hot(default_indices, 12, on_value=value_on, off_value=value_off)

            # the logits given by the trained model
            self.eval_logits = tf.placeholder(dtype=tf.float32, shape=[None, 12])
            # the correct one hot probability of output
            self.eval_output = tf.placeholder(dtype=tf.int32, shape=[None, 12])

            # get max value for each input
            max_values, max_indices = tf.nn.top_k(self.eval_logits, k=1)
            self.max_indices = max_indices

            # convert this to one hot
            one_hot = tf.gather_nd(self.one_hot_default, indices=max_indices, name='one_hot')
            self.one_hot = one_hot

            # get elementwise comparison of eval_output and one_hot for predictions
            one_hot_eq = tf.equal(one_hot, self.eval_output)
            self.one_hot_eq = one_hot_eq
            #self.one_hot = one_hot

            # get single true false value
            eq_row = tf.reduce_all(one_hot_eq, axis=1)
            self.eq_row = eq_row

            # convert true false to scalar 1 0 and sum to get no of correct preds
            correct = tf.reduce_sum(tf.where(eq_row, tf.fill(tf.shape(eq_row), 1), tf.fill(tf.shape(eq_row), 0)))
            self.correct = correct

            # add batch acc to total
            self.acc_upd = self.acc.assign_add(correct)

            # gather output one hot vectors that were correctly predicted
            indices_correct = tf.where(eq_row)
            self.indices_correct = indices_correct
            one_hot_correct = tf.gather(self.eval_output, indices_correct)
            self.one_hot_correct = one_hot_correct

            # add the gathered vectors to get no of correct predictions per
            # category in current batch
            correct_cat = tf.reduce_sum(one_hot_correct, axis=0)
            self.correct_cat = correct_cat

            # add cat batch acc to total
            self.cat_acc_upd = self.cat_acc.assign_add(correct_cat)

            self.acc_reset = tf.assign(self.acc, tf.constant(0))
            self.cat_acc_reset = tf.assign(self.cat_acc, tf.zeros(shape=(1,12), dtype=tf.int32))

    def eval_once(self, sess, ftmodel):
        sess.run([self.acc_reset, self.cat_acc_reset])

        total_acc = 0
        #cat_acc = np.asarray([[0 for i in range(12)]])
        #total_batches = ceil(5)
        total_batches = ceil(len(self.valid_input)/self.batch_size_valid)
        start_time = time.time()
        for i in range(total_batches):
            # get valid set as a batch
            bs_input = self.valid_input[i*self.batch_size_valid:(i+1)*self.batch_size_valid]
            bs_output = np.asarray(self.valid_output[i*self.batch_size_valid:(i+1)*self.batch_size_valid])
            '''print('#debug sob#')
            print('#debug bs_output: ', bs_output)'''
            # load embeddings for input and get predicted output (cur_logits)
            bs_input_ft = utils.get_ft(bs_input, ftmodel)
            input, cur_logits = sess.run([self.input, self.logits], feed_dict={self.input:bs_input_ft})
            doh, mi, oh, oheq, eqr, c, acc, ind, ohc, cc, cau =  sess.run([self.one_hot_default, self.max_indices, self.one_hot, self.one_hot_eq, self.eq_row, self.correct, self.acc_upd, self.indices_correct, self.one_hot_correct, self.correct_cat, self.cat_acc_upd], feed_dict={self.eval_logits:cur_logits, self.eval_output:bs_output})
            #oh, mi = sess.run([self.one_hot, self.mi], feed_dict={self.eval_logits:cur_logits, self.eval_output:bs_output})
            total_acc += c
            '''print('#debug mi: ', mi)
            print('#debug doh: ', doh)
            print('#debug oheq: ', oheq)
            print('#debug oh: ', oh)
            print('#debug eq_row: ', eqr)
            print('#debug corr: ', c)
            print('#debug acc_upd: ', acc)
            print('#debug ind: ', ind)
            print('#debug ohc: ', ohc)
            print('#debug cc: ', cc)
            print('#debug cau: ', cau)
            print('# debuf offense: ', acc)
            print('#debug eob#')'''
            #print('#debug cau: ', cau)
            #print('#debug oheq: ', oheq)
            #curacc = sess.run([self.acc_batch], feed_dict={self.eval_logits:cur_logits, self.eval_output:bs_output})
            #cat_acc = cau
        print('#debug time to eval: ', time.time()-start_time)
        print('#debug got total correct {} out of {}: '.format(acc, self.size_valid))
        #print(self.acc.eval())
        print('#debug correct by category: ', cau)

        print("#debug % acc: {}".format(acc/self.size_valid))

    def summary(self):
            with tf.name_scope('summaries'):
                self.loss_summary = tf.summary.scalar('loss', self.loss)
                total_acc_summary = tf.summary.scalar('total_acc', self.acc)
                # accuracy by category
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

    def build(self):
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.evaluate()
        self.summary()

    def train(self, num_epochs):
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/flair')
        writer = tf.summary.FileWriter('./graphs/flair', tf.get_default_graph())

        with tf.Session() as sess:
            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/flair/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            sess.run(tf.global_variables_initializer())

            step = self.gstep.eval()
            step_acc = self.gstep_acc.eval()

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

                        clear_input = np.zeros_like(bs_input_ft)
                        #input, logits, _, loss, loss_acc = sess.run([self.input, self.logits, self.opt, self.loss, self.loss_summary], feed_dict={self.input:bs_input_ft, self.output:bs_output})
                        #input, logits, _, last_output, loss, loss_acc, seq_output, probab, ans = sess.run([self.input, self.logits, self.opt, self.last_output, self.loss, self.loss_summary, self.seq_output, self.probs, self.output], feed_dict={self.input:bs_input_ft, self.output:bs_output})
                        input, logits, _, last_output, loss, loss_acc, seq_output, probab, ans = sess.run([self.input, self.logits, self.opt, self.last_output, self.loss, self.loss_summary, self.seq_output, self.probs, self.output], feed_dict={self.input:clear_input, self.output:bs_output})
                        print('#debug loss: ', loss)
                        '''print('#debug logits: ', logits)
                        print('#debug logits shape: ', logits.shape)
                        print('#debug log prob: ', probab)
                        print('#debug log prob shape: ', probab.shape)
                        print('#debug output: ', ans)'''
                        #loss_time = time.time()
                        #print('#debug loss time: ', loss_time-end_time)
                        #print('#debug last_output: ', last_output)
                        #print('#debug shape: ', last_output.shape)
                        #print('#debug seq_output: ', seq_output)
                        #print('#debug seq shape: ', seq_output.shape)

                        total_loss += loss
                        #print(loss_acc)
                        writer.add_summary(loss_acc, global_step=step)
                        #print(np.asarray(input).shape)
                        #print('#debug input: ', input)

                        #print(logits.shape)
                        #print('#debug output: ', logits)
                        step += 1
                        if i % 10 == 0:
                            print('#debug loss: ', loss)
                self.eval_once(sess, ftmodel)
                summary_acc = sess.run([self.summary_acc])
                #print(summary_acc[0])
                writer.add_summary(summary_acc[0], global_step=step_acc)
                step_acc += 1

                #summary = sess.run([self.summary_op])
                #writer.add_summary(summaries, global_step=step)

                print('time to train one epoch {} : {}'.format(j, time.time()-start_time))
                print("average loss at epoch {} : {}".format(j, total_loss/total_batches))
                #time.sleep(10)


if __name__ == '__main__':
    # set random seeds
    seed = 10
    np.random.seed(seed)
    tf.set_random_seed(seed)

    model = FlairPredict()
    model.build()
    model.train(25)
    print('#debug: model built!')
