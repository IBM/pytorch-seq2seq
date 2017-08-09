import os
import unittest

import torch

from seq2seq.models import DecoderRNN

class TestDecoderRNN(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.vocab_size = 100

    def test_input_dropout_WITH_PROB_ZERO(self):
        rnn = DecoderRNN(self.vocab_size, 50, 16, 0, 1, input_dropout_p=0)
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)
        output1, _, _ = rnn()
        output2, _, _ = rnn()
        for prob1, prob2 in zip(output1, output2):
            self.assertTrue(torch.equal(prob1.data, prob2.data))

    def test_input_dropout_WITH_NON_ZERO_PROB(self):
        rnn = DecoderRNN(self.vocab_size, 50, 16, 0, 1, input_dropout_p=0.5)
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)

        equal = True
        for _ in range(50):
            output1, _, _ = rnn()
            output2, _, _ = rnn()
            if not torch.equal(output1[0].data, output2[0].data):
                equal = False
                break
        self.assertFalse(equal)

    def test_dropout_WITH_PROB_ZERO(self):
        rnn = DecoderRNN(self.vocab_size, 50, 16, 0, 1, dropout_p=0)
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)
        output1, _, _ = rnn()
        output2, _, _ = rnn()
        for prob1, prob2 in zip(output1, output2):
            self.assertTrue(torch.equal(prob1.data, prob2.data))

    def test_dropout_WITH_NON_ZERO_PROB(self):
        rnn = DecoderRNN(self.vocab_size, 50, 16, 0, 1, n_layers=2, dropout_p=0.5)
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)

        equal = True
        for _ in range(50):
            output1, _, _ = rnn()
            output2, _, _ = rnn()
            if not torch.equal(output1[0].data, output2[0].data):
                equal = False
                break
        self.assertFalse(equal)
