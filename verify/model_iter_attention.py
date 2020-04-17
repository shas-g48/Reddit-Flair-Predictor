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
        self.lr = 1e-3
        self.gstep = tf.get_variable('gstep', dtype=tf.int32, initializer=tf.constant(0))
        self.gstep_acc = tf.get_variable('gstep_acc', dtype=tf.int32, initializer=tf.constant(0))
        #self.en = utils.load_vectors('data/vocab.vec')
        self.batch_size = 16    # batch size for training
        self.batch_size_valid = 500 # big batch size reduces evaluation time
        self.skip_step = 500    # to print loss at regular intervals
        #self.size = 69422    # training dataset size
        #self.size_valid = 17366
        #self.en_vec = tf.get_variable('en_vec', dtype=tf.float32,initializer=tf.constant(list(self.en.values())))
        self.init_state = tf.get_variable('init_state', dtype=tf.float32, shape = (self.batch_size, 1000), initializer=tf.zeros_initializer())

    def get_data(self):
        with tf.name_scope('data'):
            self.train_input, self.train_output = utils.load_data('data/data_correct_med.txt')
            self.valid_input, self.valid_output = utils.load_data('data/data_correct_med.txt')
            self.size = len(self.train_input)
            self.size_valid = len(self.valid_input)
            #print(self.train_input)
            #print(self.train_output)
            #self.input = tf.placeholder(dtype=tf.float32, shape=[None, 100, 300])
            #self.input = tf.placeholder(dtype=tf.float32, shape=[None, 50, 300])
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, 25, 50])
            self.output = tf.placeholder(dtype=tf.int32, shape=[None, 12])

    def inference(self):
        with tf.name_scope('inference'):
            #cell_f = tf.nn.rnn_cell.GRUCell(30)
            #cell_b = tf.nn.rnn_cell.GRUCell(15)
            cell = tf.nn.rnn_cell.GRUCell(30)
            #hidden_sizes = [12, 7, 11]
            #layers = [tf.nn.rnn_cell.GRUCell(size) for size in hidden_sizes]
            #cell = tf.nn.rnn_cell.MultiRNNCell(layers)
            # seq_output is of dims (bs, seq_len, hidden_units)
            #seq_output, out_state = tf.nn.bidirectional_dynamic_rnn(cell_f, cell_b, self.input, dtype=tf.float32)
            seq_output, out_state = tf.nn.dynamic_rnn(cell, self.input, dtype=tf.float32)#, initial_state=self.init_state)
            self.last_output = seq_output[:,24,:]   # (bs, hidden_size)

            self.seq_output = seq_output
            self.out_state = out_state
            #self.concat_output = tf.concat(seq_output,2)
            #self.last_output = self.concat_output[:,24,:]
            #self.last_output = seq_output[:,99,:]
            # output dims should be (bs, 12), input has dims (bs, hidden_units)
            # disabled for attention
            #self.logits = tf.layers.dense(self.last_output, 12, activation=None)
            #bias_logits = [1.2658978942202606, 1.3579852257480152, 1.7962401566791706, 2.8882079026286114, 2.361042985190914, 2.9551776328319828, 5.027343208116932, 3.7500284794334178, 4.343022210185258, 3.7096965608847454, 7.815875580753849, 3.693360141565052]
            #self.int_layer = tf.layers.dense(self.last_output, 60, activation=tf.nn.tanh)
            #self.logits = tf.layers.dense(self.last_output, 12, activation=None)
            #self.logits = tf.layers.dense(self.int_layer, 12, activation=None)

            # weird attention implementation
            other_output = seq_output[:,:24,:]
            self.other_output = other_output

            similarity_0 = tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,0,:]))))
            similarity_1 = tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,1,:]))))
            similarity_2 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,2,:]))))
            similarity_3 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,3,:]))))
            similarity_4 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,4,:]))))
            similarity_5 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,5,:]))))
            similarity_6 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,6,:]))))
            similarity_7 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,7,:]))))
            similarity_8 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,8,:]))))
            similarity_9 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,9,:]))))
            similarity_10 = tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,10,:]))))
            similarity_11 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,11,:]))))
            similarity_12 = tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,12,:]))))
            similarity_13 = tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,13,:]))))
            similarity_14 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,14,:]))))
            similarity_15 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,15,:]))))
            similarity_16 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,16,:]))))
            similarity_17 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,17,:]))))
            similarity_18 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,18,:]))))
            similarity_19 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,19,:]))))
            similarity_20 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,20,:]))))
            similarity_21 =  tf.transpose(tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,21,:]))))
            similarity_22 = tf.transpose( tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,22,:]))))
            similarity_23 = tf.transpose( tf.linalg.diag_part(tf.matmul(out_state, tf.transpose(other_output[:,23,:]))))

            self.similarity_23 = similarity_23
            similarity = tf.stack([similarity_0,similarity_1,similarity_2,similarity_3,similarity_4,similarity_5,similarity_6,similarity_7,similarity_8,similarity_9,similarity_10,
                                        similarity_11,similarity_12,similarity_13,similarity_14,similarity_15,similarity_16,similarity_17,similarity_18,similarity_19,similarity_20,
                                        similarity_21,similarity_22,similarity_23], axis=1)

            self.similarity = similarity
            exp_similarity = tf.exp(similarity)
            self.exp_similarity = exp_similarity

            sum_similarity = tf.reduce_sum(exp_similarity, axis=1)
            self.sum_similarity = sum_similarity

            att_score = tf.transpose(tf.divide(tf.transpose(exp_similarity), tf.transpose(sum_similarity)))
            self.att_score = att_score

            att_score_verify = tf.reduce_sum(att_score, axis=1)
            self.att_score_verify = att_score_verify

            #normalized_0 = tf.matmul(att_score[:,0], other_output[:,0,:])
            normalized_0 = tf.transpose(tf.transpose(other_output[:,0,:]) * att_score[:,0])
            self.normalized_0 = normalized_0
            normalized_1 = tf.transpose(tf.transpose(other_output[:,1,:]) * att_score[:,1])
            normalized_2 = tf.transpose(tf.transpose(other_output[:,2,:]) * att_score[:,2])
            normalized_3 = tf.transpose(tf.transpose(other_output[:,3,:]) * att_score[:,3])
            normalized_4 = tf.transpose(tf.transpose(other_output[:,4,:]) * att_score[:,4])
            normalized_5 = tf.transpose(tf.transpose(other_output[:,5,:]) * att_score[:,5])
            normalized_6 = tf.transpose(tf.transpose(other_output[:,6,:]) * att_score[:,6])
            normalized_7 = tf.transpose(tf.transpose(other_output[:,7,:]) * att_score[:,7])
            normalized_8 = tf.transpose(tf.transpose(other_output[:,8,:]) * att_score[:,8])
            normalized_9 = tf.transpose(tf.transpose(other_output[:,9,:]) * att_score[:,9])
            normalized_10 = tf.transpose(tf.transpose(other_output[:,10,:]) * att_score[:,10])
            normalized_11 = tf.transpose(tf.transpose(other_output[:,11,:]) * att_score[:,11])
            normalized_12 = tf.transpose(tf.transpose(other_output[:,12,:]) * att_score[:,12])
            normalized_13 = tf.transpose(tf.transpose(other_output[:,13,:]) * att_score[:,13])
            normalized_14 = tf.transpose(tf.transpose(other_output[:,14,:]) * att_score[:,14])
            normalized_15 = tf.transpose(tf.transpose(other_output[:,15,:]) * att_score[:,15])
            normalized_16 = tf.transpose(tf.transpose(other_output[:,16,:]) * att_score[:,16])
            normalized_17 = tf.transpose(tf.transpose(other_output[:,17,:]) * att_score[:,17])
            normalized_18 = tf.transpose(tf.transpose(other_output[:,18,:]) * att_score[:,18])
            normalized_19 = tf.transpose(tf.transpose(other_output[:,19,:]) * att_score[:,19])
            normalized_20 = tf.transpose(tf.transpose(other_output[:,20,:]) * att_score[:,20])
            normalized_21 = tf.transpose(tf.transpose(other_output[:,21,:]) * att_score[:,21])
            normalized_22 = tf.transpose(tf.transpose(other_output[:,22,:]) * att_score[:,22])
            normalized_23 = tf.transpose(tf.transpose(other_output[:,23,:]) * att_score[:,23])

            att_out = tf.add_n([normalized_0,normalized_1,normalized_2,normalized_3,normalized_4,normalized_5,normalized_6,normalized_7, normalized_8, normalized_9, normalized_10, normalized_11, normalized_12, normalized_13, normalized_14, normalized_15, normalized_16,  normalized_17, normalized_18,  normalized_19,  normalized_20, normalized_21, normalized_22, normalized_23])
            self.att_out = att_out

            concat_out = tf.concat((self.last_output,att_out), 1)
            self.concat_out = concat_out

            self.logits = tf.layers.dense(concat_out, 12, activation=None)


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
                self.loss_avg_summary = tf.summary.scalar('loss_avg', self.loss_avg)

    def loss_avg(self):
        self.total_loss = tf.placeholder(dtype=tf.float32, shape=())
        self.total_batches = tf.placeholder(dtype=tf.float32, shape=())
        self.loss_avg = self.total_loss / self.total_batches

    def build(self):
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.evaluate()
        self.loss_avg()
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

                        #clear_input = np.zeros_like(bs_input_ft)
                        #input, logits, _, loss, loss_acc = sess.run([self.input, self.logits, self.opt, self.loss, self.loss_summary], feed_dict={self.input:bs_input_ft, self.output:bs_output})
                        #print('#debug bs_input_ft: ', bs_input_ft)
                        #print('#debug bs_input_ft shape: ', bs_input_ft.shape)

                        # bidir rnn
                        #input, logits, _, last_output, loss, loss_acc, seq_output, probab, ans, concat_output = sess.run([self.input, self.logits, self.opt, self.last_output, self.loss, self.loss_summary, self.seq_output, self.probs, self.output, self.concat_output], feed_dict={self.input:bs_input_ft, self.output:bs_output})
                        # disabled to check bidir rnn
                        #input, logits, _, last_output, loss, loss_acc, seq_output, probab, ans, out_state = sess.run([self.input, self.logits, self.opt, self.last_output, self.loss, self.loss_summary, self.seq_output, self.probs, self.output, self.out_state], feed_dict={self.input:bs_input_ft, self.output:bs_output})

                        # attention
                        input, logits, _, last_output, loss, loss_acc, seq_output, probab, ans, out_state, other_output, similarity, exp_similarity, sum_similarity, att_score, similarity_23, att_score_verify, normalized_0, att_out, concat_out = sess.run([self.input, self.logits, self.opt, self.last_output, self.loss, self.loss_summary, self.seq_output, self.probs, self.output, self.out_state, self.other_output, self.similarity, self.exp_similarity, self.sum_similarity, self.att_score, self.similarity_23, self.att_score_verify, self.normalized_0, self.att_out, self.concat_out],
                        feed_dict={self.input:bs_input_ft, self.output:bs_output})
                        #input, logits, _, last_output, loss, loss_acc, seq_output, probab, ans = sess.run([self.input, self.logits, self.opt, self.last_output, self.loss, self.loss_summary, self.seq_output, self.probs, self.output], feed_dict={self.input:clear_input, self.output:bs_output})
                        #print('#debug loss: ', loss)
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
                        '''print('#debug seq_output: ', seq_output)
                        print('#debug concat_output: ', concat_output)
                        print('#debug concat_output shape: ', concat_output.shape)
                        print('#debug last_output: ', last_output)
                        print('#debug last_output shape: ', last_output.shape)'''
                        #print('#debug out_state: ', out_state)
                        #print('#debug out_state shape: ', out_state.shape)
                        '''print('#debug other_output: ', other_output)
                        print('#debug other_output shape: ', other_output.shape)
                        print('#debug similarity: ', similarity)
                        print('#debug similarity shape: ', similarity.shape)
                        print('#debug single similarity: ', similarity_23.shape)
                        print('#debug exp similarity: ', exp_similarity)
                        print('#debug exp similarity shape: ', exp_similarity.shape)
                        print('#debug sum similarity: ', sum_similarity)
                        print('#debug sum similarity shape: ', sum_similarity.shape)
                        print('#debug att score: ', att_score)
                        print('#debug att score shape: ', att_score.shape)
                        print('#debug att score verify: ', att_score_verify)
                        print('#debug att score verify shape: ', att_score_verify.shape)
                        print('#debug normalized_0 shape:', normalized_0.shape)
                        print('#debug normalized_0: ', normalized_0)
                        print('#debug concat_out: ', concat_out)
                        print('#debug concat_out shape: ', concat_out.shape)
                        print('#debug att_out shape: ', att_out.shape)
                        print('#debug att_out: ', att_out)'''
                        #print('#debug seq_output shpae: ', seq_output.shape)
                        #print(logits.shape)
                        #print('#debug output: ', logits)
                        step += 1
                        if i % 10 == 0:
                            print('#debug loss: ', loss)
                self.eval_once(sess, ftmodel)
                summary_acc = sess.run([self.summary_acc])
                #print(summary_acc[0])

                summary_loss_avg = sess.run([self.loss_avg_summary], feed_dict={self.total_loss:total_loss, self.total_batches:total_batches})
                writer.add_summary(summary_loss_avg[0], global_step=step_acc)
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
    model.train(50)
    print('#debug: model built!')
