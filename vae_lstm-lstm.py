from __future__ import print_function, division, absolute_import
import tensorflow as tf
import numpy as np

import zhusuan as zs
from zhusuan import reuse

import utils.data as data_
import utils.model as model
from utils.ptb import reader
from utils import parameters
from utils.beam_search import beam_search

from tensorflow.python import debug as tf_debug
from tensorflow.python.util.nest import flatten


# PTB input from tf tutorial
class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)

def rnn_placeholders(state):
    """Convert RNN state tensors to placeholders with the zero state as default."""
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        c, h = state
        c = tf.placeholder_with_default(c, c.shape, c.op.name)
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return tf.contrib.rnn.LSTMStateTuple(c, h)
    elif isinstance(state, tf.Tensor):
        h = state
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return h
    else:
        structure = [rnn_placeholders(x) for x in state]
        return tuple(structure)

def online_inference(sess, data_dict, sample, seq, in_state=None,
                     out_state=None, seed='<BOS>', length=None):
    """Generate sequence one word at a time, based on the previous word."""
    sentence = [seed]
    state = None
    for _ in range(params.gen_length):
        # generate until <EOS> tag
        if "<EOS>" in sentence:
            break
        input_sent_vect = [data_dict.word2idx[word] for word in sentence]
        feed = {seq: np.array(input_sent_vect).reshape([1, len(input_sent_vect)]),
                length: [len(input_sent_vect)]}
        # for the first decoder step, the state is None
        if state is not None:
             feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        sentence += [data_dict.idx2word[int(index)]]
    sentence = ' '.join([word for word in sentence if word not in ['<BOS>',
                                                                  '<EOS>']])
    print(sentence)

def q_net(encoder_input, seq_len, batch_size):
    with zs.BayesianNet() as encoder:
        # construct lstm
        # cell = tf.nn.rnn_cell.BasicLSTMCell(params.cell_hidden_size)
        # cells = tf.nn.rnn_cell.MultiRNNCell([cell]*params.rnn_layers)
        if params.base_cell == 'lstm':
          base_cell = tf.contrib.rnn.LSTMCell
        else:
          base_cell = tf.contrib.rnn.GRUCell
        cell = model.make_rnn_cell([params.decoder_hidden for _ in range(
            params.decoder_rnn_layers)], base_cell=base_cell)
        initial = cell.zero_state(batch_size, dtype=tf.float32)
        if params.keep_rate < 1:
            encoder_input = tf.nn.dropout(encoder_input, params.keep_rate)
        outputs, final_state = tf.nn.dynamic_rnn(cell,
                                                 inputs=encoder_input,
                                                 sequence_length=seq_len,
                                                 initial_state=initial,
                                                 swap_memory=True,
                                                 dtype=tf.float32)
        final_state = tf.concat(final_state[0], 1)
        if params.encode == 'hw':
            # Higway network [S.Sementiuta et.al]
            for i in range(params.highway_lc):
                with tf.variable_scope("hw_layer_enc{0}".format(i)) as scope:
                    if i == 0:  # first, input layer
                        prev_y = tf.layers.dense(final_state, params.highway_ls)
                    elif i == params.highway_lc - 1:  # last, output layer
                        final_state = tf.layers.dense(prev_y,
                                                      params.latent_size * 2)
                    else:  # hidden layers
                        prev_y = model.highway_network(prev_y,
                                                       params.highway_ls)
            lz_mean, lz_logstd = tf.split(final_state, 2, axis=1)
        elif params.encode == 'mlp':
            lz_mean = tf.layers.dense(inputs=final_state,
                                      units=params.latent_size)
            lz_logstd = tf.layers.dense(inputs=final_state,
                                        units=params.latent_size)
        # define latent variable`s Stochastic Tensor
        z = zs.Normal('z', lz_mean, lz_logstd, group_event_ndims=1)
        tf.summary.histogram('latent_space', z)
        return z

@reuse('decoder')
def vae_lstm(observed, batch_size, d_seq_l, embed, d_inputs, vocab_size, gen_mode=False):
    with zs.BayesianNet(observed=observed) as decoder:
        # prepare input
        z_mean = tf.zeros([batch_size, params.latent_size])
        z = zs.Normal('z', mean=z_mean, std=0.1, group_event_ndims=0)
        tf.summary.histogram('z|x', z)
        # z = [bath_size, l_s] -> [batch_size, seq_len, l_s]
        with tf.device("/cpu:0"):
            dec_inps = tf.nn.embedding_lookup(embed, d_inputs)
        # turn off dropout for generation:
        if params.dec_keep_rate < 1 and not gen_mode:
            dec_inps = tf.nn.dropout(dec_inps, params.dec_keep_rate)
        max_sl = tf.shape(dec_inps)[1]
        # define cell
        if params.base_cell == 'lstm':
          base_cell = tf.contrib.rnn.LSTMCell
        else:
          # not working for now
          base_cell = tf.contrib.rnn.GRUCell
        cell = model.make_rnn_cell([
          params.decoder_hidden for _ in range(
            params.decoder_rnn_layers)], base_cell=base_cell)
        if params.decode == 'hw':
            # Higway network [S.Sementiuta et.al]
            for i in range(params.highway_lc):
                with tf.variable_scope("hw_layer_enc{0}".format(i)) as scope:
                    if i == 0:  # first, input layer
                        prev_y = tf.layers.dense(z,
                                                 params.decoder_hidden * 2)
                    elif i == params.highway_lc - 1:  # last, output layer
                        z_dec = tf.layers.dense(prev_y,
                                                params.decoder_hidden * 2)
                    else:  # hidden layers
                        prev_y = model.highway_network(prev_y,
                                                       params.highway_ls)
            inp_h, inp_c = tf.split(z_dec, 2, axis=1)
            initial_state = rnn_placeholders(
                (tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h), ))
        elif params.decode == 'concat':
            z_out = tf.reshape(
              tf.tile(tf.expand_dims(z, 1), (1, max_sl, 1)),
              [batch_size, -1, params.latent_size])
            dec_inps = tf.concat([dec_inps, z_out], 2)
            initial_state = rnn_placeholders(
                cell.zero_state(tf.shape(dec_inps)[0], tf.float32))
        elif params.decode == 'mlp':
            # z->decoder initial state
            w1 = tf.get_variable('whl', [params.latent_size, params.highway_ls],
                                 tf.float32,
                                 initializer=tf.truncated_normal_initializer())
            b1 = tf.get_variable('bhl', [params.highway_ls], tf.float32,
                                 initializer=tf.ones_initializer())
            z_dec = tf.matmul(z, w1) + b1
            inp_h, inp_c = tf.split(tf.layers.dense(z_dec,
                                                    params.decoder_hidden * 2),
                                    2, axis=1)
            initial_state = rnn_placeholders(
                (tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h), ))
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=dec_inps,
                                                 sequence_length=d_seq_l,
                                                 initial_state=initial_state,
                                                 swap_memory=True,
                                                 dtype=tf.float32)
        # define decoder network
        if gen_mode:
                # only interested in the last output
                outputs = outputs[:, -1, :]
        outputs_r = tf.reshape(outputs, [-1, params.decoder_hidden])
        x_logits = tf.layers.dense(outputs_r, units=vocab_size, activation=None)
        if params.beam_search:
            sample = tf.nn.softmax(x_logits)
        else:
            sample = tf.multinomial(x_logits / params.temperature, 1)[0][0]
        return x_logits, (initial_state, final_state), sample

def main(params):
    if params.input == 'GOT':
        corpus_path = "/home/luoyy/datasets_small/got"
        data_raw = data_.got_read(corpus_path)
        data, labels_arr, embed_arr, data_dict = data_.prepare_data(data_raw,
                                                                    params)
    elif params.input == 'PTB':
        # data in form [data, labels]
        train_data_raw, valid_data_raw, test_data_raw = data_.ptb_read(
            './PTB_DATA/data')
        data, labels_arr, embed_arr, data_dict = data_.prepare_data(
            train_data_raw, params)
    with tf.Graph().as_default() as graph:
        inputs = tf.placeholder(shape=[None, None], dtype=tf.int32)
        d_inputs_ps = tf.placeholder(dtype=tf.int32, shape=[None, None])
        labels = tf.placeholder(shape=[None, None], dtype=tf.int32)
        with tf.device("/cpu:0"):
            if not params.pre_trained_embed:
                embedding = tf.get_variable(
                    "embedding", [data_dict.vocab_size,
                                  params.embed_size], dtype=tf.float32)
                vect_inputs = tf.nn.embedding_lookup(embedding, inputs)
            else:
                # [data_dict.vocab_size, params.embed_size]
                embedding = tf.Variable(
                    embed_arr,
                    trainable=params.fine_tune_embed,
                    name="embedding", dtype=tf.float32)
                vect_inputs = tf.nn.embedding_lookup(embedding, inputs)
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        vocab_size = data_dict.vocab_size
        seq_length = tf.placeholder_with_default([0.0], shape=[None])
        d_seq_length = tf.placeholder(shape=[None], dtype=tf.float32)
        qz = q_net(vect_inputs, seq_length, params.batch_size)
        x_logits, _, _ = vae_lstm({'z': qz}, params.batch_size,
                                  d_seq_length, embedding,
                                  d_inputs_ps, vocab_size=vocab_size)
        # loss, masking <PAD>
        current_len = tf.placeholder_with_default(params.sent_max_size,
                                                  shape=())
        # tf.sequence_mask, tf.contrib.seq2seq.sequence_loss
        labels_flat = tf.reshape(labels, [-1])
        cross_entr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=x_logits, labels=labels_flat)
        mask_labels = tf.sign(tf.to_float(labels_flat))
        masked_losses = mask_labels * cross_entr
        # reshape again
        masked_losses = tf.reshape(masked_losses, tf.shape(labels))
        mean_loss_by_example = tf.reduce_sum(masked_losses,
                                             reduction_indices=1) / d_seq_length
        rec_loss = tf.reduce_mean(mean_loss_by_example)
        perplexity = tf.exp(rec_loss)
        # kl divergence calculation
        kld = -0.5 * tf.reduce_mean(tf.reduce_sum(
            1 + (1e-8 + tf.square(qz.distribution.logstd))
            - tf.square(qz.distribution.mean)
            - tf.exp(tf.square(qz.distribution.logstd)), 0))
        tf.summary.scalar('kl_divergence', kld)
        # kld weight annealing
        anneal = tf.placeholder(tf.int32)
        annealing = (tf.tanh((tf.to_float(anneal) - 3500)/1000) + 1)/2
        # overall loss reconstruction loss - kl_regularization
        lower_bound = rec_loss + tf.multiply(
            tf.to_float(annealing), tf.to_float(kld)) / 10
        #lower_bound = rec_loss
        sm2 = [tf.summary.scalar('lower_bound', lower_bound),
               tf.summary.scalar('kld_coeff', annealing)]
        gradients = tf.gradients(lower_bound, tf.trainable_variables())
        opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        clipped_grad, _ = tf.clip_by_global_norm(gradients, 5)
        optimize = opt.apply_gradients(zip(clipped_grad,
                                           tf.trainable_variables()))
        #sample
        logits, states, smpl = vae_lstm({}, 1, d_seq_length, embedding,
                                        d_inputs_ps, vocab_size=vocab_size,
                                        gen_mode=True)
        init_state = states[0]
        fin_output = states[1]
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
            iters, kld_arr, coeff = [], [], []
            for e in range(params.num_epochs):
                for it in range(num_iters):
                    params.is_training = True
                    batch = data[it * params.batch_size: (it + 1) * params.batch_size]
                    l_batch = labels_arr[it * params.batch_size:(it + 1) * params.batch_size]
                    # zero padding
                    pad = len(max(batch, key=len))
                    # not optimal!!
                    length_ = np.array([len(sent) for sent in batch]).reshape(params.batch_size)
                    # prepare encoder and decoder inputs to feed
                    batch = np.array([sent + [0] * (pad - len(sent)) for sent in batch])
                    l_batch = np.array([(sent + [0] * (pad - len(sent))) for sent in l_batch])
                    # encoder feed=[....<EOS>], decoder feed=[<BOS>....], labels=[.....<EOS>]
                    feed = {inputs: l_batch, d_inputs_ps: batch, labels: l_batch,
                            seq_length: length_, d_seq_length: length_, anneal: cur_it, current_len: pad}
                    lb, _, kld_, ann_, r_loss, perplexity_ = sess.run([lower_bound, optimize,
                                                                       kld, annealing, rec_loss, perplexity],
                                                                      feed_dict=feed)
                    cur_it += 1
                    iters.append(cur_it)
                    kld_arr.append(kld_)
                    coeff.append(ann_)
                    if cur_it % 100 == 0 and cur_it != 0:
                        print("VLB after {} ({}) iterations (epoch): {} KLD: "
                              "{} Annealing Coeff: {} CE: {}".format(
                                  cur_it, e,lb, kld_, ann_, r_loss))
                        print("Perplexity: {}".format(perplexity_))
                    if cur_it % 150 == 0:
                        if not params.beam_search:
                            params.is_training = False
                            online_inference(sess, data_dict,
                                             sample=smpl, seq=d_inputs_ps,
                                             in_state=init_state,
                                             out_state=fin_output,
                                             length=d_seq_length)
                        else:
                            gen_sentence = beam_search(sess, data_dict, states,
                                                       smpl, (d_inputs_ps,
                                                        d_seq_length), params,
                                                       beam_size=params.beam_size)
                            print(gen_sentence)
                    if cur_it % 400 == 0 and cur_it!=0:
                       # saver = tf.train.Saver()
                        summary = sess.run(merged, feed_dict=feed)
                        summary_writer.add_summary(summary)
                        # saver.save(sess, os.path.join(params.LOG_DIR, "lstmlstm_model.ckpt"), cur_it)
                    if params.visualise:
                        if cur_it % 30000 == 0 and cur_it!=0:
                           import matplotlib.pyplot as plt
                           with open("./run_kld" + str(params.dec_keep_rate), 'w') as wf:
                               _ = [wf.write(str(s) + ' ')for s in iters]
                               wf.write('\n')
                               _ = [wf.write(str(s) + ' ')for s in kld_arr]
                               wf.write('\n')
                               _ = [wf.write(str(s) + ' ') for s in coeff]
                           plt.plot(iters, kld_arr, label='KLD')
                           plt.xlabel('Iterations')
                           plt.legend(bbox_to_anchor=(1.05, 1),
                                      loc=1, borderaxespad=0.)
                           plt.show()
                           plt.plot(iters, coeff, 'r--', label='annealing')
                           plt.legend(bbox_to_anchor=(1.05, 1),
                                      loc=1, borderaxespad=0.)
                           plt.show()
if __name__=="__main__":
    params = parameters.Parameters()
    params.parse_args()
    main(params)
