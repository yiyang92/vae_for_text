from __future__ import print_function, division, absolute_import
import tensorflow as tf
import numpy as np

import zhusuan as zs
from zhusuan import reuse

import utils.data as data_
import beam_search as bs
import utils.model as model
from utils.ptb import reader
from utils import parameters

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

def online_inference(sess, data_dict, sample, seq, in_state=None, out_state=None, seed='<BOS>', length=None):
    """ Generate sequence one word at a time, based on the previous word
    """
    sentence = [seed]
    state = None
    for _ in range(params.gen_length):
        # generate until <EOS> tag
        if "<EOS>" in sentence:
            break
        input_sent_vect = [data_dict.word2idx[word] for word in sentence]
        feed = {seq: np.array(input_sent_vect).reshape([1, len(input_sent_vect)]), length: [len(input_sent_vect)]}
        # for the first decoder step, the state is None
        if state is not None:
             feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        sentence += [data_dict.idx2word[int(index)]]
    sentence = ' '.join([word for word in sentence if word not in ['<BOS>',
                                                                  '<EOS>']])
    print(sentence)

def q_net(x, seq_len, batch_size):
    with zs.BayesianNet() as encoder:
        # construct lstm
        # cell = tf.nn.rnn_cell.BasicLSTMCell(params.cell_hidden_size)
        # cells = tf.nn.rnn_cell.MultiRNNCell([cell]*params.rnn_layers)
        cell = model.make_rnn_cell([params.decoder_hidden for _ in range(params.decoder_rnn_layers)], base_cell=params.base_cell)
        initial = cell.zero_state(batch_size, dtype=tf.float32)
        print(int)
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
                    prev_y = model.highway_network(prev_y, params.highway_ls)

        outputs, final_state = tf.nn.dynamic_rnn(cell,
                                                 inputs=encoder_input,
                                                 sequence_length=seq_len,
                                                 initial_state=initial,
                                                 swap_memory=True,
                                                 dtype=tf.float32)
        final_state = tf.concat(final_state[0], 1)
        lz_mean = tf.layers.dense(inputs=final_state, units=params.latent_size, activation=None)
        lz_logstd = tf.layers.dense(inputs=final_state, units=params.latent_size, activation=None)
        # define latent variable`s Stochastic Tensor
        z = zs.Normal('z', lz_mean, lz_logstd, group_event_ndims=1)
        tf.summary.histogram('latent_space', z)
        return z

@reuse('decoder')
def vae_lstm(observed, batch_size, d_seq_l, embed, d_inputs, vocab_size, dropout_off=False):
    with zs.BayesianNet(observed=observed) as decoder:
        # prepare input
        z_mean = tf.zeros([batch_size, params.latent_size])
        z_logstd = tf.zeros([batch_size, params.latent_size])
        z = zs.Normal('z', mean=z_mean, logstd=z_logstd, group_event_ndims=0)
        tf.summary.histogram('z|x', z)
        # z = [bath_size, l_s] -> [batch_size, seq_len, l_s]
        with tf.device("/cpu:0"):
            dec_inps = tf.nn.embedding_lookup(embed, d_inputs)
        # turn off dropout for generation:
        if params.dec_keep_rate < 1 and not dropout_off:
            dec_inps = tf.nn.dropout(dec_inps, params.dec_keep_rate)
        max_sl = tf.shape(dec_inps)[1]
        z_out = tf.reshape(tf.tile(tf.expand_dims(z, 1), (1, max_sl, 1)), [batch_size, -1, params.latent_size])
        c_inputs = tf.concat([dec_inps, z_out], 2)
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
        cell = model.make_rnn_cell([
          params.decoder_hidden for _ in range(
            params.decoder_rnn_layers)], base_cell=params.base_cell)
        initial_state = rnn_placeholders(
            (tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h), ))
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=c_inputs,
                                                 sequence_length=d_seq_l,
                                                 initial_state=initial_state,
                                                 swap_memory=True,
                                                 dtype=tf.float32)
        # define decoder network
        outputs_r = tf.reshape(outputs, [-1, params.decoder_hidden])
        x_logits = tf.layers.dense(outputs_r, units=vocab_size, activation=None)
        # take unnormalized log-prob of the last word in sequence and sample from multinomial distibution
        sample = None
        norm_log_prob = None
        if params.beam_search:
            logits_ = tf.reshape(x_logits, [tf.shape(outputs)[0],
                                            tf.shape(outputs)[1],
                                            vocab_size])[:, -1]
            top_k = tf.nn.top_k(logits_, params.beam_size)
            sample = top_k.indices
            norm_log_prob = tf.log(tf.nn.softmax(top_k.values))
        sample_gr = tf.multinomial(tf.reshape(x_logits, [tf.shape(outputs)[0],
                                                         tf.shape(outputs)[1],
                                                         vocab_size])[:, -1]
                                    /params.temperature, 1)[:, 0][:]
        return decoder, x_logits, initial_state, final_state, sample_gr, sample, norm_log_prob

# TODO: print values of input and decoder output
def main(params):
    if params.input == 'GOT':
        corpus_path = "/home/luoyy/datasets_small/got"
        if not params.pre_trained_embed:
            data_raw, labels = data_.tokenize_text_and_make_labels(corpus_path)
        else:
            w2_vec, data_raw = data_.load_word_embeddings(corpus_path, params.input, params.embed_size)
        # get embeddings, prepare data
        # TODO: change initial data process for no pretrained embedding option
        print("building dictionary")
        data_dict = data_.Dictionary(data_raw)
        print(data_raw[1])
        data = [data_dict.seq2dx(dt[1:]) for dt in data_raw if len(dt) < params.sent_max_size]
        labels_arr = [data_dict.seq2dx(dt[:-1]) for dt in data_raw if len(dt) < params.sent_max_size]
        print("----Corpus_Information--- \n Raw data size: {} sentences \n Vocabulary size {}"
              "\n Limited data size {} sentences \n".format(len(data_raw), data_dict.vocab_size, len(data)))
        if params.pre_trained_embed:
            embed_arr = np.zeros([data_dict.vocab_size, params.embed_size])
            for i in range(embed_arr.shape[0]):
                if i == 0:
                    continue
                embed_arr[i] = w2_vec.word_vec(data_dict.idx2word[i])
    elif params.input == 'PTB':
        # data in form [data, labels]
        train_data_raw, valid_data_raw, test_data_raw = data_.ptb_read('./PTB_DATA/data')
        print(train_data_raw[0:2])
        data_dict = data_.Dictionary(train_data_raw)
        print("----Corpus_Information--- \n Train data size: {} sentences \n Vocabulary size {}"
              "\n Test data size {}".format(len(train_data_raw), data_dict.vocab_size, len(test_data_raw)))
        # raw data ['<BOS>'...'<EOS>']
        if params.pre_trained_embed:
            w2_vec, data_raw = data_.load_word_embeddings("PTB", params.input, params.embed_size, w2vec_it=5,
                                                          sentences=train_data_raw, tokenize=False)
            embed_arr = np.zeros([data_dict.vocab_size, params.embed_size])
            for i in range(embed_arr.shape[0]):
                if i == 0:
                    continue
                embed_arr[i] = w2_vec.word_vec(data_dict.idx2word[i])
        # data=[<BOS> ....], labels=[.....<EOS>]
        data = [[data_dict.word2idx[word] for word in sent[:-1]] for sent in train_data_raw]
        labels_arr = [[data_dict.word2idx[word] for word in sent[1:]] for sent in train_data_raw]
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
        decoder, dec_logits, initial_state, final_state, _, _, _,= vae_lstm({'z': qz},
                                                                           params.batch_size, d_seq_length,
                                                                           embedding, d_inputs_ps, vocab_size=vocab_size)
        # loss, masking <PAD>
        current_len = tf.placeholder_with_default(params.sent_max_size, shape=())
        # tf.sequence_mask, tf.contrib.seq2seq.sequence_loss
        labels_flat = tf.reshape(labels, [-1])
        cross_entr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=dec_logits, labels=labels_flat)
        mask_labels = tf.sign(tf.to_float(labels_flat))
        masked_losses = mask_labels * cross_entr
        # reshape again
        masked_losses = tf.reshape(masked_losses, tf.shape(labels))
        mean_loss_by_example = tf.reduce_sum(masked_losses,
                                             reduction_indices=1) / d_seq_length
        rec_loss = tf.reduce_mean(mean_loss_by_example)
        perplexity = tf.exp(rec_loss)
        # kl divergence calculation
        kld = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + qz.distribution.logstd
                                                  - tf.square(qz.distribution.mean)
                                                  - tf.exp(qz.distribution.logstd), 1))
        tf.summary.scalar('kl_divergence', kld)
        # kld weight annealing
        anneal = tf.placeholder(tf.int32)
        #annealing = tf.sigmoid((tf.to_float(anneal) - 2500)/100 + 1)
        annealing = (tf.tanh((tf.to_float(anneal) - 3500)/1000) + 1)/2
        # overall loss reconstruction loss - kl_regularization
        lower_bound = tf.reduce_mean(
          d_seq_length)*rec_loss + tf.multiply(
            tf.to_float(annealing), tf.to_float(kld))
        #lower_bound = rec_loss
        sm2 = [tf.summary.scalar('lower_bound', lower_bound),
               tf.summary.scalar('kld_coeff', annealing)]
        gradients = tf.gradients(lower_bound, tf.trainable_variables())
        opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        clipped_grad, _ = tf.clip_by_global_norm(gradients, 1)
        optimize = opt.apply_gradients(zip(clipped_grad, tf.trainable_variables()))
        #sample
        _, logits, init_state, fin_output, smpl, sample,norm_log, = vae_lstm({}, 1, d_seq_length, embedding, d_inputs_ps, vocab_size=vocab_size, dropout_off=True)
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
                        print("VLB after {} ({}) iterations (epoch): {} KLD: {} Annealing Coeff: {} CE: {}".format(cur_it, e,lb, kld_, ann_, r_loss))
                        print("Perplexity: {}".format(perplexity_))
                    if cur_it % 150 == 0:
                        if not params.beam_search:
                            params.is_training = False
                            online_inference(sess, data_dict, sample=smpl, seq=d_inputs_ps, in_state=init_state,
                                         out_state=fin_output, length=d_seq_length)
                        else:
                            gen_sentence = bs.beam_search(sess, d_inputs_ps, data_dict, norm_log, sample, init_state, fin_output,
                                            params.gen_length, length=d_seq_length, seed='<BOS>')
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
