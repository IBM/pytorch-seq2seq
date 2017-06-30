import os
import unittest

import torch

from seq2seq.evaluator import Predictor
from seq2seq.dataset import Dataset
from seq2seq.models import Seq2seq, EncoderRNN, DecoderRNN

class TestPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_wd = os.getcwd()
        self.dataset = Dataset(path=os.path.join(self.test_wd,'tests/data/eng-fra.txt'),
                               src_max_len=50, tgt_max_len=50, src_max_vocab=50000, tgt_max_vocab=50000)
        self.encoder = EncoderRNN(self.dataset.input_vocab,max_len=10, hidden_size=10, rnn_cell='lstm')
        self.decoder = DecoderRNN(self.dataset.output_vocab, max_len=10, hidden_size=10, rnn_cell='lstm')
        self.seq2seq = Seq2seq(self.encoder,self.decoder)
        if torch.cuda.is_available():
            self.seq2seq.cuda()
        self.mock_seq2seq = Seq2seq(self.encoder, self.decoder)

        for param in self.seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    def test_predict(self):
        predictor = Predictor(self.seq2seq,
                self.dataset.input_vocab, self.dataset.output_vocab)
        src_seq = ["I", "am", "fat"]
        tgt_seq = predictor.predict(src_seq)
        for tok in tgt_seq:
            self.assertTrue(tok in self.dataset.output_vocab._token2index)
