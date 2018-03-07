
import os
import tqdm
import nltk
import multiprocessing
import pickle
import numpy as np
import collections


def got_read(corpus_path, sent_file="./trained_embeddings/sent_got.pickle"):

    if os.path.exists(sent_file):
        print("Loading sentences file")
        with open(sent_file, 'rb') as rf:
            sentences = pickle.load(file=rf)
        return sentences

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
        sentences.append(
            ['<BOS>'] + nltk.word_tokenize(sent) + ['<EOS>'])
    with open(sent_file, 'wb') as wf:
        pickle.dump(sentences, file=wf)
    return sentences

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

def load_sentences(corpus_path):
    if tokenize:
        assert (sentences is None),\
        'Tokenize option cannot be used wth provided sentences'
        sentences, _ = tokenize_text_and_make_labels(corpus_path)
        sentences = [['<BOS>'] + dt + ['<EOS>'] for dt in sentences]
    elif tokenize:
        assert (sentences is None), "Must provide sentences"
    return sentences

def train_w2vec(embed_fn, embed_size, w2vec_it=5, tokenize=True,
                sentences=None, model_path="./trained_embeddings"):
    from gensim.models import KeyedVectors, Word2Vec
    embed_fn += '.embed'
    print(os.path.join(model_path, embed_fn))
    print("Corpus contains {0:,} tokens".format(
        sum(len(sent) for sent in sentences)))
    if os.path.exists(os.path.join(model_path, embed_fn)):
        print("Loading existing embeddings file")
        return KeyedVectors.load_word2vec_format(
            os.path.join(model_path, embed_fn))
    # sample parameter-downsampling for frequent words
    w2vec = Word2Vec(sg=0,
                     workers=multiprocessing.cpu_count(),
                     size=embed_size, min_count=0, window=5, iter=w2vec_it)
    w2vec.build_vocab(sentences=sentences)
    print("Training w2vec")
    w2vec.train(sentences=sentences,
                total_examples=w2vec.corpus_count, epochs=w2vec.iter)
    # Save it to model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    w2vec.wv.save_word2vec_format(os.path.join(model_path, embed_fn))
    return KeyedVectors.load_word2vec_format(os.path.join(model_path, embed_fn))

class Dictionary(object):
    def __init__(self, sentences, vocab_drop):
        # sentences - array of sentences
        self._vocab_drop = vocab_drop
        if vocab_drop < 0:
            raise ValueError
        self._sentences = sentences
        self._word2idx = {}
        self._idx2word = {}
        self._words = []
        self.get_words()
        # add tokens
        #self._words.append('<EOS>')
        #self._words.append('<BOS>')
        self._words.append('<unk>')
        self.build_vocabulary()
        self._mod_sentences()

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
                word = word if word in ["<EOS>",
                                        "<BOS>",
                                        "<PAD>", "<UNK>",
                                        "N"] else word.lower()
                self._words.append(word)

    def _mod_sentences(self):
        # for every sentence, if word not in vocab set to <unk>
        for i in range(len(self._sentences)):
            sent = self._sentences[i]
            for j in range(len(sent)):
                try:
                    self.word2idx[sent[j]]
                except:
                    sent[j] = '<unk>'
            self._sentences[i] = sent

    def build_vocabulary(self):
        counter = collections.Counter(self._words)
        # words, that occur less than 5 times dont include
        sorted_dict = sorted(counter.items(), key= lambda x: (-x[1], x[0]))
        # keep n words to be included in vocabulary
        sorted_dict = [(wd, count) for wd, count in sorted_dict
                       if count >= self._vocab_drop or wd in ['<unk>',
                                                              '<BOS>',
                                                              '<EOS>']]
        # after sorting the dictionary, get ordered words
        words, _ = list(zip(*sorted_dict))
        self._word2idx = dict(zip(words, range(1, len(words) + 1)))
        self._idx2word = dict(zip(range(1, len(words) + 1), words))
        # add <PAD> as zero
        self._idx2word[0] = '<PAD>'
        self._word2idx['<PAD>'] = 0

    def __len__(self):
        return len(self.idx2word)


def extract_files(files_dir, extract_to):
    import tarfile
    if not os.path.exists(extract_to):
        for file in os.listdir(files_dir):
            tarfile.open(os.path.join(files_dir, file), 'r:gz').extractall(extract_to)
    return 'Extraction successful'
