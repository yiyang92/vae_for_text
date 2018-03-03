import tensorflow as tf
import numpy as np


def beam_search(sess, inputs_ps, data_dict, log_prob, sample, state_ps, out_state, gen_seq_len, length,seed='<BOS>',
                beam_size=2):
    pass
import heapq

# from tf/models/im2txt
class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
          heapq.heappush(self._data, x)
        else:
          heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
          data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []

# from tf/models/im2txt
class Beam(object):
    """Used for beam_search"""
    def __init__(self, sentence, state, logprob, score):
        self.sentence = sentence
        self.logprob = logprob
        self.state = state
        self.score = score

    def __cmp__(self, other):
        """Compares captions by score."""
        assert isinstance(other, Beam)
        if self.score == other.score:
          return 0
        elif self.score < other.score:
          return -1
        else:
          return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Beam)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Beam)
        return self.score == other.score

def beam_search(sess, data_dict, states, sample, seq_len, params,
                beam_size=2, ret_beams=False, len_norm_f=0.7, seed='<BOS>'):
        """Generate text using beam search.
        Args:
            sess: tf.Session
            states: tuple (in_state, out_state)
            sample: normalized probabilities of the next word
            seq_len: tuple (input_ps, length)
            params: parameters class instance
            beam_size: keep how many beam candidates
            len_norm_f: beam search length normalization parameter
        Returns:
            best_beam: the best beam, contains text with the biggest
        sum of probabilities
        """
        try:
            seed = data_dict.word2idx[seed]
        except:
            raise ValueError("seed word is not found in the vocabulary, "
                             "use <BOS>")
            seed = data_dict.word2idx['<BOS>']
        stop_word = data_dict.word2idx['<EOS>']
        # state placeholder and ops
        in_state, out_state = states
        gen_list = {'text': ' '}
        # initial feed
        #input_sent_vect = [data_dict.word2idx[word] for word in sentence]
        seq, length = seq_len
        feed = {seq: np.array(seed).reshape([1, 1]),
                length: [1]}
        # probs are normalized probs
        probs, state = sess.run([sample, out_state], feed)
        # initial Beam, pushed to the heap (TopN class)
        # inspired by tf/models/im2txt
        initial_beam = Beam(sentence=[seed],
                            state=state,
                            logprob=0.0,
                            score=0.0)
        partial_captions = TopN(beam_size)
        partial_captions.push(initial_beam)
        complete_captions = TopN(beam_size)
        # continue to generate, until max_len
        for _ in range(params.gen_length - 1):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()
            # get last word in the sentence
            input_feed = [(c.sentence[-1],
                           len(c.sentence)) for c in partial_captions_list]
            state_feed = [c.state for c in partial_captions_list]
            # get states and probs for every beam
            probs_list, states_list = [], []
            for inp_length, state in zip(input_feed, state_feed):
                inp, len_ = inp_length
                feed = {seq: np.array(inp).reshape([1, 1]),
                        length: [len_],
                        in_state: state}
                probs, new_state = sess.run([sample, out_state], feed)
                probs_list.append(probs)
                states_list.append(new_state)
            # for every beam get candidates and append to list
            for i, partial_caption in enumerate(partial_captions_list):
                cur_state = states_list[i]
                cur_probs = probs_list[i]
                # sort list probs, enumerate to remember indices (I like python "_")
                w_probs = list(enumerate(cur_probs.ravel()))
                w_probs.sort(key=lambda x: -x[1])
                # keep n probs
                w_probs = w_probs[:beam_size]
                for w, p in w_probs:
                    if p < 1e-12:
                        continue  # Avoid log(0).
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + np.log(p)
                    score = logprob
                    # complete caption, got <EOS>
                    if w == stop_word:
                        if len_norm_f > 0:
                            score /= len(sentence)**len_norm_f
                        beam = Beam(sentence, cur_state, logprob, score)
                        complete_captions.push(beam)
                    else:
                        beam = Beam(sentence, cur_state, logprob, score)
                        partial_captions.push(beam)
            if partial_captions.size() == 0:
                # When all captions are complete
                break
        # If we have no complete captions then fall back to the partial captions.
        # But never output a mixture of complete and partial captions because a
        # partial caption could have a higher score than all the complete captions.
        if not complete_captions.size():
            complete_captions = partial_captions
        # find the best beam
        beams = complete_captions.extract(sort=True)
        if not ret_beams:
            best_beam = beams[0]
            capt = [data_dict.idx2word[word] for word in best_beam.sentence
                    if word not in [seed, stop_word]]
            gen_list['text'] = ' '.join(capt)
        # return list of beam candidates
        if ret_beams:
            c_list = []
            for c in beams:
                capt = [data_dict.idx2word[word] for word in c.sentence
                        if word not in [seed, stop_word]]
                c_list.append(' '.join(capt))
            gen_list['text'] = c_list
        return gen_list['text']
