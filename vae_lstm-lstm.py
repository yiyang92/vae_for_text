from __future__ import print_function, division, absolute_import

import tensorflow as tf

from tensorflow.contrib import layers
import zhusuan as zs
from zhusuan import reuse
import utils.data as data_


import utils.model as model
from utils.ptb import reader
from tensorflow.python import debug as tf_debug

import numpy as np

import os

# PTB input from tf tutorial
class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class Parameters():
    encoder_hidden = 191  # std=191, inputless_dec=350
    decoder_hidden = 191
    rnn_layers = 1
    decoder_rnn_layers = 1
    batch_size = 40
    latent_size = 13  # std=13, inputless_dec=111
    embed_size = 353 # std=353, inputless_dec=499
    num_epochs = 40
    learning_rate = 0.00001
    sent_max_size = 300
    base_cell = tf.contrib.rnn.GRUCell
    temperature = 0.5
    gen_length = 10
    keep_rate = 1.0
    dec_keep_rate = 0.62
    highway_lc = 4
    highway_ls = embed_size
    is_training = True
    LOG_DIR = './model_logs/'
    datasets = ['GOT', 'PTB']
    input = datasets[1]
    debug = False

params = Parameters()

def online_inference(sess, data_dict, sample, seq, in_state=None, out_state=None, seed='<EOS>', length=[1]):
    """ Generate sequence one character at a time, based on the previous character
    """
    sentence = [seed]
    state = None
    for _ in range(params.gen_length):
        input_sent_vect = [data_dict.word2idx[word] for word in sentence]
        feed = {seq: np.array(input_sent_vect).reshape([1, len(input_sent_vect)]), length: [len(input_sent_vect)]}
        # for the first decoder step, the state is None
        if state is not None:
             feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        # print("Index: ", index)
        # print("Index shape", index.shape)
        # print(type(sample))
        sentence += [data_dict.idx2word[int(index)]]
    print([word for word in sentence if word not in ['<EOS>', '<BOS>']])


def q_net(x, seq_len, batch_size=params.batch_size):
    with zs.BayesianNet() as encoder:
        # construct lstm
        # cell = tf.nn.rnn_cell.BasicLSTMCell(params.cell_hidden_size)
        # cells = tf.nn.rnn_cell.MultiRNNCell([cell]*params.rnn_layers)
        cell = model.make_rnn_cell([params.decoder_hidden for _ in range(params.decoder_rnn_layers)], base_cell=params.base_cell)
        initial = cell.zero_state(batch_size, dtype=tf.float32)
        if params.keep_rate < 1:
            x = tf.nn.dropout(x, params.keep_rate)
        s_l = tf.shape(x)[1]
        # Higway network [S.Sementiuta et.al]
        for i in range(params.highway_lc):
            with tf.variable_scope("hw_layer_enc{0}".format(i)) as scope:
                if i == 0:  # first, input layer
                    x = tf.reshape(x, [-1, params.embed_size])
                    prev_y = tf.layers.dense(x, params.highway_ls, tf.nn.relu)
                elif i == params.highway_lc - 1:  # last, output layer
                    encoder_input = tf.layers.dense(prev_y, params.embed_size)
                    encoder_input = tf.reshape(encoder_input, [params.batch_size, s_l, params.embed_size])
                else:  # hidden layers
                    print(i)
                    prev_y = model.highway_network(prev_y, params.highway_ls)

        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=encoder_input, sequence_length=seq_len,
                                                initial_state=initial, swap_memory=True, dtype=tf.float32)
        final_output = tf.reshape(outputs[:, -1, :], [batch_size, -1])
        tf.summary.histogram('encoder_out', final_output)
        lz_mean = tf.layers.dense(inputs=final_state[0], units=params.latent_size, activation=None)
        lz_logstd = tf.layers.dense(inputs=final_state[0], units=params.latent_size, activation=None)
        # define latent variable`s Stochastic Tensor
        z = zs.Normal('z', lz_mean, lz_logstd, group_event_ndims=1)
        tf.summary.histogram('latent_space', z)
        return z


@reuse('decoder')
def vae_lstm(observed, batch_size, d_seq_l, embed, d_inputs, vocab_size):
    with zs.BayesianNet(observed=observed) as decoder:
        # prepare input
        z_mean = tf.zeros([batch_size, params.latent_size])
        z_logstd = tf.zeros([batch_size, params.latent_size])
        z = zs.Normal('z', mean=z_mean, logstd=z_logstd, group_event_ndims=1)
        tf.summary.histogram('z|x', z)
        # z = [bath_size, l_s] -> [batch_size, seq_len, l_s]
        with tf.device("/cpu:0"):
            dec_inps = tf.nn.embedding_lookup(embed, d_inputs)
        if params.dec_keep_rate < 1 and params.is_training:
            dec_inps = tf.nn.dropout(dec_inps, params.dec_keep_rate)
        max_sl = tf.shape(dec_inps)[1]
        z_out = tf.reshape(tf.tile(tf.expand_dims(z, 1), (1, max_sl, 1)), [batch_size, -1, params.latent_size])
        c_inputs = tf.concat([dec_inps, z_out], 2)
        w1 = tf.get_variable('whl', [params.latent_size, params.highway_ls], tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        b1 = tf.get_variable('bhl', [params.highway_ls], tf.float32, initializer=tf.ones_initializer())
        z_dec = tf.nn.relu(tf.matmul(z, w1) + b1)
        inp_state = tf.layers.dense(z_dec, params.decoder_hidden)
        cell = model.make_rnn_cell([params.decoder_hidden for _ in range(params.decoder_rnn_layers)], base_cell=params.base_cell)
        #print(cell.zero_state(params.batch_size, dtype=tf.float32))
        # input cell.zero_state(tf.shape(inputs)[0], dtype=tf.float32)
        initial_state = tf.placeholder_with_default(input=tf.expand_dims(inp_state, 0),
                                          shape=[None, None, params.decoder_hidden])
        ins = tf.reshape(initial_state, [-1, params.decoder_hidden])
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=c_inputs, sequence_length=d_seq_length,
                                                 initial_state=(ins, ), swap_memory=True, dtype=tf.float32)
        # define decoder network
        x_logits = tf.layers.dense(outputs, units=vocab_size)
        print("x_logits",x_logits)
        x = zs.Categorical('x', logits=tf.exp(x_logits)/params.temperature, group_event_ndims=1)
        sample = tf.squeeze(x[:, -1])
        return decoder, x_logits, initial_state, final_state, sample

# TODO: print values of input and decoder output
if __name__ == "__main__":
    if params.input == 'GOT':
        corpus_path = "/home/luoyy/datasets_small/got"
        data_raw, labels = data_.tokenize_text_and_make_labels(corpus_path)
        # get embeddings, prepare data
        print("building dictionary")
        data_dict = data_.Dictionary(data_raw)
        print(data_raw[1])
        data = [data_dict.seq2dx(dt) for dt in data_raw if len(dt) < params.sent_max_size]
        data = [dt + data_dict.seq2dx(['<EOS>']) for dt in data]
        labels_arr = [data_dict.seq2dx(dt) for dt in data_raw if len(dt) < params.sent_max_size]
        print("----Corpus_Information--- \n Raw data size: {} sentences \n Vocabulary size {}"
              "\n Limited data size {} sentences".format(len(data_raw), data_dict.vocab_size, len(data)))
    elif params.input == 'PTB':
        # data in form [data, labels]
        train_data_raw, valid_data_raw, test_data_raw = data_.ptb_read('./PTB_DATA/data')
        # Can test and validationa data contain more words??
        print(len(train_data_raw))
        print(train_data_raw[0:2])
        data_dict = data_.Dictionary(train_data_raw)
        print("----Corpus_Information--- \n Train data size: {} sentences \n Vocabulary size {}"
              "\n Test data size {}".format(len(train_data_raw), data_dict.vocab_size, len(test_data_raw)))
        # raw data ['<BOS>'...'<EOS>']
        # TODO: use test dataset for perplexity calculation
        data = [[data_dict.word2idx[word] for word in sent[:-1]] for sent in train_data_raw]
        labels_arr = [[data_dict.word2idx[word] for word in sent[1:]] for sent in train_data_raw]
    with tf.Graph().as_default() as graph:
        inputs = tf.placeholder(shape=[None, None], dtype=tf.int32)
        d_inputs_ps = tf.placeholder(dtype=tf.int32, shape=[None, None])
        labels = tf.placeholder(shape=[None, None], dtype=tf.int64)
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [data_dict.vocab_size, params.embed_size], dtype=tf.float32)
            vect_inputs = tf.nn.embedding_lookup(embedding, inputs)
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        vocab_size = data_dict.vocab_size
        seq_length = tf.placeholder(shape=[None], dtype=tf.float32)
        d_seq_length = tf.placeholder(shape=[None], dtype=tf.float32)

        qz = q_net(vect_inputs, seq_length)
        decoder, dec_logits, initial_state, final_state, sample = vae_lstm({'z': qz},
                                                                           params.batch_size, d_seq_length,
                                                                           embedding, d_inputs_ps, vocab_size=vocab_size)
        # loss
        log_px_z = decoder.local_log_prob('x')
        rec_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=dec_logits))
        # kl divergence calculation
        kld = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + qz.distribution.logstd
                                                  - tf.square(qz.distribution.mean)
                                                  - tf.exp(qz.distribution.logstd), 1))
        tf.summary.scalar('kl_divergence', kld)
        # kld weight annealing
        anneal = tf.placeholder(tf.int32)
        annealing = tf.sigmoid((tf.to_float(anneal) - 4500)/100 + 1)
        # overall loss reconstruction loss - kl_regularization
        lower_bound = 79 * rec_loss + tf.multiply(tf.to_float(annealing), tf.to_float(kld))
        sm2 = tf.summary.scalar('lower_bound', lower_bound)
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        optimize = optimizer.minimize(lower_bound)

        #sample
        _, logits, init_state, fin_output, smpl = vae_lstm({}, 1, d_seq_length, embedding, d_inputs_ps, vocab_size=vocab_size)
        # sample = tf.multinomial(tf.exp(logits[:, -1] / params.temperature), 1)[:, 0]
        # merge summaries
        merged = tf.summary.merge_all()
        with tf.Session() as sess:

            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer()])

            if params.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            summary_writer = tf.summary.FileWriter(params.LOG_DIR, sess.graph)
            summary_writer.add_graph(sess.graph)
            #ptb_data = PTBInput(params.batch_size, train_data)
            num_iters = len(data) // params.batch_size
            cur_it = 0
            for e in range(params.num_epochs):
                for it in range(num_iters):
                    params.is_training = True
                    # data
                    batch = data[it * params.batch_size: (it + 1) * params.batch_size]
                    # zero padding
                    pad = len(max(batch, key=len))
                    # not optimal!!
                    length_ = np.array([len(sent) for sent in batch]).reshape(params.batch_size)
                    batch = np.flip(np.array([sent + [0] * (pad - len(sent)) for sent in batch]), 1)
                    l_batch = labels_arr[it * params.batch_size:(it + 1) * params.batch_size]
                    l_batch = np.array([(sent + [0] * (pad - len(sent))) for sent in l_batch])
                    feed = {inputs: batch, d_inputs_ps: l_batch, labels: l_batch,
                            seq_length: length_, d_seq_length: length_, anneal: cur_it}
                    lb, _, kld_, ann_, r_loss = sess.run([lower_bound, optimize, kld, annealing, rec_loss ], feed_dict=feed)
                    cur_it += 1
                    if cur_it % 100 == 0 and cur_it != 0:
                        print("VLB {} iterations: {} KLD: {} Coeff: {} CE: {}".format(cur_it, lb, kld_, ann_, r_loss))
                    if cur_it % 250 == 0  and cur_it != 0:
                        print("Variational lower bound after {} iterations: {} KLD: {}".format(cur_it, lb, kld_))
                        params.is_training = False
                        online_inference(sess, data_dict, sample=smpl, seq=d_inputs_ps, in_state=init_state,
                                         out_state=fin_output, length=d_seq_length)
                    if cur_it % 400 == 0 and cur_it!=0:
                       pass
                       # saver = tf.train.Saver()
                       # summary = sess.run([merged], feed_dict=feed)
                       # summary_writer.add_summary(summary)
                        # saver.save(sess, os.path.join(params.LOG_DIR, "lstmlstm_model.ckpt"), cur_it)


