from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import tqdm

import utils.data as data_
import utils.model as model

params = {
    'batch_size': 20,
    'num_epochs': 20,
    'embed_size': 500,
    'num_hidden': 1100,
    'num_layers': 1,
    'learning_rate': 0.001,
    'mode_train': True,
    'sent_max_size': 228,
    'gen_length': 20,
    'temperature': 0.5,
    'keep_rate': 0.6,
    'input': ['GOT', 'PTB'][0]
}


def online_inference(sess, data_dict, sample, seq, in_state=None, out_state=None, seed='king'):
    """ Generate sequence one character at a time, based on the previous character
    """
    sentence = [seed]
    state = None
    for _ in range(params['gen_length']):
        input_sent_vect = [data_dict.word2idx[word] for word in sentence]
        feed = {seq: np.array(input_sent_vect).reshape([1, len(input_sent_vect)]), length: [len(input_sent_vect)]}
        # for the first decoder step, the state is None
        if state is not None:
             feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        sentence += [data_dict.idx2word[idx] for idx in index]
    print([word for word in sentence if word not in ['<EOS>', '<PAD>']])


if __name__ == "__main__":
    if params['input'] == 'GOT':
        # GOT corpus
        corpus_path = "/home/luoyy/datasets_small/got"
        data_raw, labels = data_.tokenize_text_and_make_labels(corpus_path)
        # get embeddings, prepare data
        print("building dictionary")
        data_dict = data_.Dictionary(data_raw)
        print(data_raw[1])
        data = [data_dict.seq2dx(dt) for dt in data_raw if len(dt) < params['sent_max_size']]
        labels_arr = [data_dict.seq2dx(dt) for dt in labels if len(dt) < params['sent_max_size']]
        print(labels[1])
        print("----Corpus_Information--- \n Raw data size: {} sentences \n Vocabulary size {}" 
              "\n Limited data size {} sentences".format(len(data_raw), data_dict.vocab_size,  len(data)))
        del(data_raw)
    elif params['input'] == 'PTB':
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
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [data_dict.vocab_size, params['embed_size']], dtype=tf.float32)
            vect_inputs = tf.nn.embedding_lookup(embedding, inputs)
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        if params['mode_train'] and params['keep_rate'] < 1:
            vect_inputs = tf.nn.dropout(vect_inputs, params['keep_rate'])
        labels = tf.placeholder(shape=[None, None], dtype=tf.int64)

        cell = model.make_rnn_cell([params['num_hidden']]*params['num_layers'],
                                   base_cell=tf.contrib.rnn.GRUCell)

        initial_state = tf.placeholder_with_default(input=cell.zero_state(tf.shape(vect_inputs)[0], dtype=tf.float32),
                                          shape=[None, None, params['num_hidden']])

        zs = cell.zero_state(params['batch_size'], dtype=tf.float32)
        print(initial_state)
        length = tf.placeholder(shape=[None], dtype=tf.float32)
        ins = tf.reshape(initial_state, [-1, params['num_hidden']])
        # TODO: find a way how to initialize 2-layers network
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=vect_inputs, sequence_length=length,
                                                initial_state=(ins, )*params['num_layers'], swap_memory=False,
                                                 dtype=tf.float32)
        fc_layer = tf.layers.dense(inputs=outputs, units=data_dict.vocab_size)

        prnt = tf.Print(fc_layer, [tf.shape(final_state), tf.shape(zs)])
        # define optimization with lr decay, lr decay can be use with SGD oprtimizer
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step, 500, 0.96)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=fc_layer))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # not possible to use seq2seq loss with different lengths
        predictions = tf.argmax(fc_layer, axis=2)
        corr_predictions = tf.to_float(tf.equal(labels, predictions))
        accuracy = tf.reduce_mean(corr_predictions)
        # sample from multinomial distribution
        # take [batch, seq_length, vocab_size] as input
        sample = tf.multinomial(tf.exp(fc_layer[:, -1] / params['temperature']), 1)[:, 0]
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
                    feed = {inputs: batch, labels: l_batch, length: length_, global_step: 0}
                    loss_, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=feed)
                    cur_it += 1
                    if it % 250 == 0:
                        params['mode_train'] = False
                        _ = sess.run(prnt, feed_dict=feed)
                        online_inference(sess, data_dict, sample=sample, seq=inputs, in_state=initial_state, out_state=final_state)
                        print("loss after {it} operations: {loss_}, accuracy: {acc}".format(**locals()))
