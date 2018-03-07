from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import tqdm

import utils.data as data_
import utils.model as model
from utils.parameters import Parameters

# TODO : Make multiple layers work!
params = {
    'batch_size': 20,
    'num_epochs': 20,
    'embed_size': 464,
    'num_hidden': 337,
    'num_layers': 1,
    'learning_rate': 0.001,
    'mode_train': True,
    'sent_max_size': 228,
    'gen_length': 20,
    'temperature': 0.5,
    'keep_rate': 0.66,
    'input': ['GOT', 'PTB'][1],
    'vocab_drop': 3
}
# for back compatibility
params_c = Parameters()
params_c.batch_size = params['batch_size']
params_c.num_epochs = params['num_epochs']
params_c.embed_size = params['embed_size']
params_c.learning_rate = params['learning_rate']
params_c.pre_trained_embed = False
params_c.beam_search = False
params_c.vocab_drop = params['vocab_drop']
params_c.embed_size = params['embed_size']


def online_inference(sess, data_dict, sample, seq, in_state=None, out_state=None, seed='<BOS>'):
    """ Generate sequence one character at a time, based on the previous character
    """
    sentence = [seed]
    state = None
    for _ in range(params['gen_length']):
        # generate until <EOS> tag
        if "<EOS>" in sentence:
            break
        input_sent_vect = [data_dict.word2idx[word] for word in sentence]
        feed = {seq: np.array(input_sent_vect).reshape([1,
                                                        len(input_sent_vect)]),
                length: [len(input_sent_vect)],
                keep_rate: 1.0}
        # for the first decoder step, the state is None
        if state is not None:
             feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        sentence += [data_dict.idx2word[idx] for idx in index]
    print(' '.join([word for word in sentence if word not in ['<EOS>',
                                                             '<PAD>', '<BOS>']]))

if __name__ == "__main__":
    if params['input'] == 'GOT':
        # GOT corpus
        corpus_path = "/home/luoyy/datasets_small/got"
        data_raw = data_.got_read(corpus_path)
        data, labels_arr, _, data_dict = data_.prepare_data(data_raw,
                                                                    params_c)
        vocab_size = data_dict.vocab_size
        print("Most common words : {}", [data_dict.idx2word[i] for i in range(vocab_size - 1, vocab_size - 7, -1)])
        del(data_raw)
    elif params['input'] == 'PTB':
        # data in form [data, labels]
        train_data_raw, valid_data_raw, test_data_raw = data_.ptb_read('./PTB_DATA/data')
        # data in form [data, labels]
        train_data_raw, valid_data_raw, test_data_raw = data_.ptb_read(
            './PTB_DATA/data')
        data, labels_arr, _, data_dict = data_.prepare_data(
            train_data_raw, params_c)
    with tf.Graph().as_default() as graph:
        inputs = tf.placeholder(shape=[None, None], dtype=tf.int32)
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [data_dict.vocab_size, params['embed_size']], dtype=tf.float32)
            vect_inputs = tf.nn.embedding_lookup(embedding, inputs)
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        keep_rate = tf.placeholder(tf.float32)
        if params['mode_train'] and params['keep_rate'] < 1:
            vect_inputs = tf.nn.dropout(vect_inputs, keep_rate)

        labels = tf.placeholder(shape=[None, None], dtype=tf.int64)
        cell = model.make_rnn_cell([params['num_hidden']]*params['num_layers'],
                                   base_cell=tf.contrib.rnn.GRUCell)

        initial_state = tf.placeholder_with_default(input=cell.zero_state(tf.shape(vect_inputs)[0], dtype=tf.float32),
                                          shape=[None, None, params['num_hidden']])
        zs = cell.zero_state(params['batch_size'], dtype=tf.float32)
        length = tf.placeholder(shape=[None], dtype=tf.float32)
        ins = tf.reshape(initial_state, [-1, params['num_hidden']])
        # TODO: find a way how to initialize 2-layers network
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=vect_inputs, sequence_length=length,
                                                initial_state=(ins, )*params['num_layers'], swap_memory=False,
                                                 dtype=tf.float32)
        fc_layer = tf.layers.dense(inputs=outputs, units=data_dict.vocab_size, activation=None)
        prnt = tf.Print(fc_layer, [tf.shape(final_state), tf.shape(zs)])
        # define optimization with lr decay, lr decay can be use with SGD oprtimizer
        global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step, 500, 0.96)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fc_layer))
        # clip gradients
        gradients = tf.gradients(loss, tf.trainable_variables())
        opt = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        clipped_grad, _ = tf.clip_by_global_norm(gradients, 1)
        optimizer = opt.apply_gradients(zip(clipped_grad, tf.trainable_variables()))
        # not possible to use seq2seq loss with different lengths
        predictions = tf.argmax(fc_layer, axis=2)
        corr_predictions = tf.to_float(tf.equal(labels, predictions))
        accuracy = tf.reduce_mean(corr_predictions)
        # sample from multinomial distribution
        # take [batch, seq_length, vocab_size] as input
        sample = tf.multinomial(tf.exp(fc_layer[:, -1] / params['temperature']), 1)[:, 0][:]
        print(fc_layer)
        prnt2 = tf.Print(sample, [tf.shape(sample), fc_layer[:, -1]])

        num_iters = len(data) // params['batch_size']
        print("Number of iterations per epoch: ", num_iters)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # print(graph.get_operations())
            cur_it = 0
            for e in range(params['num_epochs']):
                for it in tqdm.trange(num_iters):
                    params['mode_train'] = True
                    batch = data[it * params['batch_size']:(it+1) * params['batch_size']]
                    # zero padding
                    pad = len(max(batch, key=len))
                    # not optimal!!
                    length_ = np.array([len(sent) for sent in batch]).reshape(params['batch_size'])
                    batch = np.array([sent+[0]*(pad - len(sent)) for sent in batch])
                    l_batch = labels_arr[it * params['batch_size']: (it+1) * params['batch_size']]
                    l_batch = np.array([(sent+[0]*(pad - len(sent))) for sent in l_batch])
                    feed = {inputs: batch, labels: l_batch, length: length_, global_step: 0,
                            keep_rate: params['keep_rate']}
                    loss_, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=feed)
                    cur_it += 1
                    if it % 250 == 0:
                        params['mode_train'] = False
                        feed[keep_rate] = 1.0
                        _ = sess.run(prnt, feed_dict=feed)
                        online_inference(sess, data_dict, sample=sample, seq=inputs, in_state=initial_state, out_state=final_state)
                        print("loss after {it} operations: {loss_}, accuracy: {acc}".format(**locals()))
