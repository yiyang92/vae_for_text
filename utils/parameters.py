class Parameters():
    # general parameters
    latent_size = 13  # std=13, inputless_dec(keep_rate=0.0)=111
    num_epochs = 150
    learning_rate = 0.001
    batch_size = 50
    # for decoding
    temperature = 1.0
    gen_length = 40
    # beam search
    beam_search = False
    beam_size = 2
    # encoder
    rnn_layers = 1
    encoder_hidden = 191  # std=191, inputless_dec=350
    keep_rate = 1.0
    highway_lc = 2
    highway_ls = 600
    # decoder
    decoder_hidden = 191
    decoder_rnn_layers = 1
    dec_keep_rate = 0.75
    # data
    datasets = ['GOT', 'PTB']
    embed_size = 353 # std=353, inputless_dec=499
    sent_max_size = 300
    input = datasets[1]
    debug = False
    # use pretrained w2vec embeddings
    pre_trained_embed = True
    fine_tune_embed = True
    # technical parameters
    is_training = True
    LOG_DIR = './model_logs/'
    visualise = False
    # gru base cell partially implemented
    import tensorflow as tf
    base_cell = tf.contrib.rnn.LSTMCell
    #base_cell = tf.contrib.rnn.GRUCell
    def parse_args(self):
        import argparse
        parser = argparse.ArgumentParser(
            description="Specify some parameters, all parameters "
            "also can be directly specified in Parameters class")
        parser.add_argument('--dataset', default=self.input,
                            help='training dataset (GOT or PTB)', dest='data')
        parser.add_argument('--ls', default=self.latent_size,
                            help='latent space size', dest='ls')
        parser.add_argument('--lr', default=self.learning_rate,
                            help='learning rate', dest='lr')
        parser.add_argument('--embed_dim', default=self.embed_size,
                            help='embedding size', dest='embed')
        parser.add_argument('--lst_state_dim_enc', default=self.encoder_hidden,
                            help='encoder state size', dest='enc_hid')
        parser.add_argument('--lst_state_dim_dec', default=self.decoder_hidden,
                            help='decoder state size', dest='dec_hid')
        parser.add_argument('--latent', default=self.latent_size,
                            help='latent space size', dest='latent')
        parser.add_argument('--dec_dropout', default=self.dec_keep_rate,
                            help='decoder dropout keep rate', dest='dec_drop')
        parser.add_argument('--beam_search', default=self.beam_search,
                            action="store_true")
        parser.add_argument('--beam_size', default=self.beam_size)

        args = parser.parse_args()
        self.input = args.data
        self.latent_size = args.ls
        self.learning_rate = args.lr
        self.embed_size = args.embed
        self.encoder_hidden = args.enc_hid
        self.decoder_hidden = args.dec_hid
        self.latent_size = args.latent
        self.dec_keep_rate = args.dec_drop
        self.beam_search = args.beam_search
        self.beam_size = args.beam_size
