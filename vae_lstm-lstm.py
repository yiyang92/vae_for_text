from __future__ import print_function, division, absolute_import

import tensorflow as tf

from tensorflow.contrib import layers
import zhusuan as zs
from zhusuan import reuse
import utils.data as data_


import utils.model as model
from utils.ptb import reader

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
    encoder_hidden = 350  # std=191, inputless_dec=350
    decoder_hidden = 350
    rnn_layers = 1
    decoder_rnn_layers = 1
    batch_size = 100
    latent_size = 111  # std=13, inputless_dec=111
    embed_size = 499 # std=353, inputless_dec=499
    num_epochs = 40
    learning_rate = 0.0001
    sent_max_size = 128
    base_cell = tf.contrib.rnn.GRUCell
    temperature = 1.0
    gen_length = 50
    keep_rate = 0.62
    dec_keep_rate = 1.0
    highway_lc = 99
    highway_ls = 50
    is_training = True
    LOG_DIR = './model_logs/'
    datasets = ['GOT', 'PTB']
    input = datasets[1]

params = Parameters()


def online_inference(sess, data_dict, sample, seq, in_state=None, out_state=None, seed='<BOS>', length=[1]):
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
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=x, sequence_length=seq_len,
                                                initial_state=initial, swap_memory=True, dtype=tf.float32)
        final_output = tf.reshape(outputs[:, -1, :], [batch_size, -1])
        tf.summary.histogram('encoder_out', final_output)
        for i in range(params.highway_lc):
            with tf.name_scope("layer{0}".format(i)) as scope:
                if i == 0:  # first, input layer
                    prev_y = tf.layers.dense(final_output, params.highway_ls, tf.nn.relu)
                elif i == params.highway_lc - 1:  # last, output layer
                    lz_mean = tf.layers.dense(inputs=prev_y, units=params.latent_size)
                    lz_logstd = tf.layers.dense(inputs=prev_y, units=params.latent_size)
                else:  # hidden layers
                    prev_y = model.highway_network(prev_y, params.highway_ls)

        # define latent variable`s Stochastic Tensor
        z = zs.Normal('z', lz_mean, lz_logstd, group_event_ndims=1)
        tf.summary.histogram('latent_space', z)
        # kl divergence calculation
        kld = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + lz_logstd - tf.square(lz_mean) - tf.exp(lz_logstd), 1))
        tf.summary.scalar('kl_divergence', kld)
        return encoder, kld


@reuse('decoder')
def vae_lstm(observed, batch_size, d_seq_l, embed, d_inputs, vocab_size):
    with zs.BayesianNet(observed=observed) as decoder:
        # prepare input
        eos_input = tf.tile(tf.reshape(tf.convert_to_tensor([data_dict.word2idx['<BOS>']]), [1, 1]),
                            (batch_size, 1))
        if params.is_training:
            d_inputs = tf.concat([eos_input, d_inputs[:, :-1]], 1)
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
        inputs = tf.concat([dec_inps, z_out], 2)
        cell = model.make_rnn_cell([params.decoder_hidden for _ in range(params.decoder_rnn_layers)], base_cell=params.base_cell)
        initial_state = tf.placeholder_with_default(input=cell.zero_state(tf.shape(inputs)[0], dtype=tf.float32),
                                          shape=[None, None, params.decoder_hidden])
        ins = tf.reshape(initial_state, [-1, params.decoder_hidden])
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=inputs, sequence_length=d_seq_l,
                                                 initial_state=(ins, ), swap_memory=True, dtype=tf.float32)
        # define decoder network
        x_logits = tf.layers.dense(outputs, units=vocab_size)
        x = zs.Categorical('x', logits=x_logits, group_event_ndims=1)
        sample = tf.squeeze(x[:, -1])
        return decoder, x_logits, initial_state, final_state, sample

# TODO: add dropout?
# TODO: what feeed to the decoder?
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

        def log_joint(observed):
            # vae_conv(observed, batch_size, d_seq_l, dec_inps, vocab_size, max_sl)
            decoder, _, _, _, _ = vae_lstm(observed, params.batch_size, d_seq_length, embedding, d_inputs_ps, vocab_size=vocab_size)
            log_pz, log_px_z = decoder.local_log_prob(['z', 'x'])
            return log_px_z + log_pz

        encoder, kld = q_net(vect_inputs, seq_length)
        q_z_outs, log_q_z = encoder.query('z', outputs=True, local_log_prob=True)
        smq = tf.summary.histogram('q(z|x)', q_z_outs)
        lower_bound = tf.reduce_mean(zs.sgvb(log_joint, observed={'x': inputs}, latent={'z': [q_z_outs, log_q_z]}))
        prnt = tf.Print(lower_bound, [tf.shape(q_z_outs), tf.shape(log_q_z)])
        sm2 = tf.summary.scalar('lower_bound', lower_bound)
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        optimize = optimizer.minimize(-lower_bound)

        #sample
        _, logits, init_state, fin_output, smpl = vae_lstm({}, 1, d_seq_length, embedding, d_inputs_ps, vocab_size=vocab_size)
        # sample = tf.multinomial(tf.exp(logits[:, -1] / params.temperature), 1)[:, 0]
        # merge summaries
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer()])

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
                    batch = np.array([sent + [0] * (pad - len(sent)) for sent in batch])
                    l_batch = labels_arr[it * params.batch_size:(it + 1) * params.batch_size]
                    l_batch = np.array([(sent + [0] * (pad - len(sent))) for sent in l_batch])
                    #elif params.input == 'PTB':
                        #batch, l_batch = reader.ptb_producer(train_data, params.batch_size, num_iters, name='train')
                    # TODO: Feed values

                    feed = {inputs: batch, d_inputs_ps: batch, labels: l_batch, seq_length: length_, d_seq_length: length_}
                    lb, _, kld_ = sess.run([lower_bound, optimize, kld], feed_dict=feed)
                    cur_it += 1
                    if cur_it % 100 == 0 and cur_it != 0:
                        print("Variational lower bound after {} iterations: {} KLD: {}".format(cur_it, lb, kld_))
                    if cur_it % 250 == 0  and cur_it != 0:
                        print("Variational lower bound after {} iterations: {} KLD: {}".format(cur_it, lb, kld_))
                        params.is_training = False
                        online_inference(sess, data_dict, sample=smpl, seq=d_inputs_ps, in_state=init_state,
                                         out_state=fin_output, length=d_seq_length)
                    if cur_it % 400 == 0:
                        saver = tf.train.Saver()
                        summary, _ = sess.run([merged, prnt], feed_dict=feed)
                        summary_writer.add_summary(summary)
                        # saver.save(sess, os.path.join(params.LOG_DIR, "lstmlstm_model.ckpt"), cur_it)


