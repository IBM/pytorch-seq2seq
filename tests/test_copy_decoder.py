import unittest

import torch
from torch.autograd import Variable

from seq2seq.models import CopyDecoder

class TestCopyDecoder(unittest.TestCase):

    def test_forward(self):
        hidden_size = 16
        output_size = 200
        decoder = CopyDecoder(hidden_size, output_size)

        batch_size = 8
        en_len = 5
        de_len = 6
        hidden = Variable(torch.randn(batch_size, de_len, hidden_size))
        attn = Variable(torch.randn(batch_size, de_len, en_len))
        output = decoder(hidden, attn)
        self.assertEquals(output.size(), torch.Size((batch_size * de_len, output_size + en_len)))
