import os
import unittest

import torch

from seq2seq.dataset import Dataset
from seq2seq.models import EncoderRNN

class TestEncoderRNN(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_wd = os.getcwd()
        self.dataset = Dataset.from_file(path=os.path.join(self.test_wd,'tests/data/eng-fra.txt'),
                               src_max_len=50, tgt_max_len=50, src_max_vocab=50000, tgt_max_vocab=50000)

    def test_input_dropout_WITH_PROB_ZERO(self):
        rnn = EncoderRNN(self.dataset.input_vocab, 50, 16, input_dropout_p=0)
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)
        batch = [[1,2,3], [1,2], [1]]
        output1, _ = rnn(batch)
        output2, _ = rnn(batch)
        self.assertTrue(torch.equal(output1[0].data, output2[0].data))

    def test_input_dropout_WITH_NON_ZERO_PROB(self):
        rnn = EncoderRNN(self.dataset.input_vocab, 50, 16, input_dropout_p=0.5)
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)
        batch = [[1,2,3], [1,2], [1]]

        equal = True
        for _ in range(50):
            output1, _ = rnn(batch)
            output2, _ = rnn(batch)
            if not torch.equal(output1[0].data, output2[0].data):
                equal = False
                break
        self.assertFalse(equal)

    def test_dropout_WITH_PROB_ZERO(self):
        rnn = EncoderRNN(self.dataset.input_vocab, 50, 16, dropout_p=0)
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)
        batch = [[1,2,3], [1,2], [1]]
        output1, _ = rnn(batch)
        output2, _ = rnn(batch)
        self.assertTrue(torch.equal(output1[0].data, output2[0].data))

    def test_dropout_WITH_NON_ZERO_PROB(self):
        rnn = EncoderRNN(self.dataset.input_vocab, 50, 16, n_layers=2, dropout_p=0.5)
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)
        batch = [[1,2,3], [1,2], [1]]

        equal = True
        for _ in range(50):
            output1, _ = rnn(batch)
            output2, _ = rnn(batch)
            if not torch.equal(output1[0].data, output2[0].data):
                equal = False
                break
        self.assertFalse(equal)
