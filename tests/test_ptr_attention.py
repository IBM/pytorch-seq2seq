import os
import unittest

import torch
from torch.autograd import Variable

from seq2seq.models.attention import PointerAttention

class TestDecoderRNN(unittest.TestCase):

    def test_shape(self):
        batch_size = 8
        input_len = 10
        output_len = 11
        hidden_size = 16

        ptr_attn = PointerAttention(hidden_size)

        output = Variable(torch.randn(batch_size, output_len, hidden_size))
        context = Variable(torch.randn(batch_size, input_len, hidden_size))

        output, attn = ptr_attn(output, context)

        self.assertEqual((batch_size, output_len, input_len), output.size())
