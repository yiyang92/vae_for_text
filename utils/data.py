
import os
import tqdm
import nltk
import multiprocessing
import pickle
import numpy as np
import collections


def tokenize_text_and_make_labels(corpus_path, sent_file="./trained_embeddings/sent_got.pickle",
                                  labels_file="./trained_embeddings/labels_got.pickle"):
    if os.path.exists(sent_file) and os.path.exists(labels_file):
        print("Loading sentences file")
        with open(sent_file, 'rb') as rf:
            sentences = pickle.load(file=rf)
        with open(labels_file, 'rb') as wlf:
            labels = pickle.load(file=wlf)
        return sentences, labels

    if not os.path.exists("./trained_embeddings"):
        os.makedirs("./trained_embeddings")
    sentences = []
    files = sorted(os.listdir(corpus_path))
    raw_vocab = u""
    for file in tqdm.tqdm(files):
        with open(corpus_path + '/' + file) as rf:
            for line in rf.readlines():
                raw_vocab += line.lower()
    # download nltk specific files
    nltk.download("punkt")
    # tokenize sentences first
    sent_raw = nltk.sent_tokenize(raw_vocab)
    # obtain list of sentences with splitted words
    sentences = []
    labels = []
    for sent in sent_raw:
        sentences.append(nltk.word_tokenize(sent))
    del sent_raw
    for sent in sentences:
        sent = sent[1:]
        sent.append('<EOS>')
        labels.append(sent)

    with open(sent_file, 'wb') as wf:
        pickle.dump(sentences, file=wf)
    with open(labels_file, 'wb') as wlf:
        pickle.dump(labels, file=wlf)
    return sentences, labels


def ptb_data_read(corpus_file, sent_file):
    if os.path.exists(sent_file):
        print("Loading sentences file")
        with open(sent_file, 'rb') as rf:
            sentences = pickle.load(file=rf)
        return sentences

    if not os.path.exists("./trained_embeddings"):
        os.makedirs("./trained_embeddings")
    sentences = []
    with open(corpus_file) as rf:
        for line in rf:
            sentences.append(['<BOS>'] + line.strip().split(' ') + ['<EOS>'])
    with open(sent_file, 'wb') as wf:
        pickle.dump(sentences, file=wf)
    return sentences


def ptb_read(data_path):
    # train_set
    train_data = ptb_data_read(os.path.join(data_path, 'ptb.train.txt'),
                                                sent_file="./trained_embeddings/sent_ptb_train.pickle")

    test_data = ptb_data_read(os.path.join(data_path, 'ptb.test.txt'),
                                                sent_file="./trained_embeddings/sent_ptb_test.pickle")

    val_data = ptb_data_read(os.path.join(data_path, 'ptb.valid.txt'),
                                                sent_file="./trained_embeddings/sent_ptb_val.pickle")

    return train_data, test_data, val_data


# use wor2vec for learning word-embeddings
def load_word_embeddings(corpus_path, embed_fn, embed_size, w2vec_it=5, tokenize=True, sentences=None, model_path="./trained_embeddings"):
    from gensim.models import KeyedVectors, Word2Vec
    embed_fn += '.embed'
    print(os.path.join(model_path, embed_fn))
    if tokenize:
        assert (sentences is None), 'Tokenize option cannot be used wth provided sentences'
        sentences, _ = tokenize_text_and_make_labels(corpus_path)
        sentences = [['<BOS>'] + dt + ['<EOS>'] for dt in sentences]
    elif tokenize:
        assert (sentences is None), "Must provide sentences"
    if os.path.exists(os.path.join(model_path, embed_fn)):
        print("Loading existing embeddings file")
        return KeyedVectors.load_word2vec_format(os.path.join(model_path, embed_fn)), sentences
    # Print corpus info, start w2vec training
    print("Corpus contains {0:,} tokens".format(sum(len(sent) for sent in sentences)))
    # :TODO integrate hardcoded parameters into class
    # sample parameter-downsampling for frequent words
    print(sentences[0:5])
    w2vec = Word2Vec(sg=0, workers=multiprocessing.cpu_count(), size=embed_size, min_count=0, window=5, iter=w2vec_it)
    w2vec.build_vocab(sentences=sentences)
    print("Training w2vec")
    w2vec.train(sentences=sentences, total_examples=w2vec.corpus_count, epochs=w2vec.iter)
    # Save it to model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    w2vec.wv.save_word2vec_format(os.path.join(model_path, embed_fn))
    return KeyedVectors.load_word2vec_format(os.path.join(model_path, embed_fn)), sentences

# TODO: implement batch generator
class BatchGenerator():
    def __init__(self):
        pass

    def next_batch(self):
        pass


class Dictionary(object):
    def __init__(self, sentences):
        # sentences - array of sentences
        self._sentences = sentences
        self._word2idx = {}
        self._idx2word = {}
        self._words = []
        self.get_words()
        # add tokens
        self._words.append('<EOS>')
        self._words.append('<BOS>')
        self.build_vocabulary()

    @property
    def vocab_size(self):
        return len(self._idx2word)

    @property
    def sentences(self):
        return self._sentences

    @property
    def word2idx(self):
        return self._word2idx

    @property
    def idx2word(self):
        return self._idx2word

    def seq2dx(self, sentence):
        return [self.word2idx[wd] for wd in sentence]

    def get_words(self):
        for sent in self.sentences:
            for word in sent:
                self._words.append(word)

    def build_vocabulary(self):
        counter = collections.Counter(self._words)
        sorted_dict = sorted(counter.items(), key= lambda x: (-x[1], x[0]))
        # after sorting the dictionary, get ordered words
        words, _ = list(zip(*sorted_dict))
        self._word2idx = dict(zip(words, range(len(words))))
        self._idx2word = dict(zip(range(len(words)), words))
        self._word2idx['<PAD>'] = 0
        self._idx2word[0] = '<PAD>'

    def __len__(self):
        return len(self.idx2word)


def extract_files(files_dir, extract_to):
    import tarfile
    if not os.path.exists(extract_to):
        for file in os.listdir(files_dir):
            tarfile.open(os.path.join(files_dir, file), 'r:gz').extractall(extract_to)
    return 'Extraction successful'

