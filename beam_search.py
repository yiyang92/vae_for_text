import tensorflow as tf
import numpy as np


def beam_search(sess, inputs_ps, data_dict, log_prob, sample, state_ps, out_state, gen_seq_len, length,seed='<BOS>',
                beam_size=2):
    seed = data_dict.word2idx[seed]
    stop_word = data_dict.word2idx['<EOS>']
    # need to get 3 highest probabilities, than for teach of them find p(x1)*p(x2|x1)...
    beam = [[seed] for _ in range(beam_size)]
    beam_prob = np.zeros([beam_size, gen_seq_len])
    # initial feed
    feed = {inputs_ps:  np.array([seed]).reshape([1, 1]),
         length: [1]}
    # to tf add log_prob returring op norm_log_prob = tf.log(tf.softmax(...))
    # sample = tf.multinomial(...)
    index_arr, state, probs = sess.run([sample, out_state, log_prob], feed)
    #print(index_arr)
    index_arr = index_arr[0]
    # keeping previous states of hypothesis sentences
    states = [state] * beam_size
    # probabilities
    probs = probs[0][:beam_size]*-1
    # append to beam
    beam_prob[:, 0] = np.ones([beam_size])
    for j in range(beam_size):
         beam[j].append(index_arr[j])
         beam_prob[j][1] = probs[j]
    for st in range(2, gen_seq_len):
        for i in range(beam_size):
            if stop_word in beam[i]:
                continue
            len_ = len(beam[i])
            feed = {inputs_ps: np.array(beam[i]).reshape([1, len_]),
                length: [len_], state_ps: states[i]}
            # feed to network get probs
            index_arr, state, probs = sess.run([sample, out_state, log_prob], feed)
            index_arr = index_arr[0]
            states[i] = state
            # probabilities
            probs = probs[0][:beam_size]*-1
            max_sum_index = 0
            # find probability max_sum_index and append to a beam
            choose_index = -1
            for j in range(beam_size):
                temp_sum = probs[j] + np.sum(beam_prob[i])
                if temp_sum > max_sum_index:
                    max_sum_index = temp_sum
                    choose_index += 1
            beam_prob[i][st] = probs[choose_index]
            beam[i].append(index_arr[choose_index])
    # find the best beam
    best_beam =beam[np.argmax(np.sum(beam_prob, 1))]
    #print("Best beam", best_beam)
    #print(beam)
    return [data_dict.idx2word[word] for word in best_beam]



