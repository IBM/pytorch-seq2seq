import os
import unittest

import torch
from torch.autograd import Variable
from seq2seq.models import EncoderRNN

class TestEncoderRNN(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.vocab_size = 100
        self.input_var = Variable(torch.randperm(self.vocab_size).view(10, 10))
        self.lengths = [10] * 10

    def test_input_dropout_WITH_PROB_ZERO(self):
        rnn = EncoderRNN(self.vocab_size, 50, 16, input_dropout_p=0)
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)
        output1, _ = rnn(self.input_var, self.lengths)
        output2, _ = rnn(self.input_var, self.lengths)
        self.assertTrue(torch.equal(output1.data, output2.data))

    def test_input_dropout_WITH_NON_ZERO_PROB(self):
        rnn = EncoderRNN(self.vocab_size, 50, 16, input_dropout_p=0.5)
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)

        equal = True
        for _ in range(50):
            output1, _ = rnn(self.input_var, self.lengths)
            output2, _ = rnn(self.input_var, self.lengths)
            if not torch.equal(output1.data, output2.data):
                equal = False
                break
        self.assertFalse(equal)

    def test_dropout_WITH_PROB_ZERO(self):
        rnn = EncoderRNN(self.vocab_size, 50, 16, dropout_p=0)
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)
        output1, _ = rnn(self.input_var, self.lengths)
        output2, _ = rnn(self.input_var, self.lengths)
        self.assertTrue(torch.equal(output1.data, output2.data))

    def test_dropout_WITH_NON_ZERO_PROB(self):
        # It's critical to set n_layer=2 here since dropout won't work
        # when the RNN only has one layer according to pytorch's doc
        rnn = EncoderRNN(self.vocab_size, 50, 16, n_layers=2, dropout_p=0.5)
        for param in rnn.parameters():
            param.data.uniform_(-1, 1)

        equal = True
        for _ in range(50):
            output1, _ = rnn(self.input_var, self.lengths)
            output2, _ = rnn(self.input_var, self.lengths)
            if not torch.equal(output1.data, output2.data):
                equal = False
                break
        self.assertFalse(equal)

    def test_pretrained_embedding(self):
        hidden_size = 16
        pretrained_embedding = torch.randn(self.vocab_size, hidden_size)
        rnn = EncoderRNN(self.vocab_size, 50, hidden_size,
                         embedding=pretrained_embedding,
                         update_embedding=False)
        self.assertTrue(torch.equal(pretrained_embedding, rnn.embedding.weight.data))
        self.assertFalse(rnn.embedding.weight.requires_grad)
