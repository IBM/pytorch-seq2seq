import os
import unittest

import torch
from torch.autograd import Variable
import torchtext

from seq2seq.data import Seq2SeqDataset
from seq2seq.models import CopyDecoder

class TestCopyDecoder(unittest.TestCase):

    def test_forward(self):
        hidden_size = 16
        batch_size = 8

        dataset = self._init_dataset()
        batch_iterator = torchtext.data.BucketIterator(
            dataset=dataset, batch_size=batch_size,
            sort_key=lambda x: -len(x.src),
            device=-1, repeat=False)

        output_size = len(dataset.fields['tgt'].vocab)
        decoder = CopyDecoder(hidden_size, output_size)
        for batch in batch_iterator:
            en_len, de_len = batch.src[0].size(1), batch.tgt.size(1)
            src_lengths = batch.src[1]
            hidden = Variable(torch.randn(batch_size, de_len, hidden_size))
            attn = Variable(torch.randn(batch_size, de_len, en_len))
            output, symbols = decoder(hidden, attn, batch, dataset)
            self.assertEquals(output.size(), torch.Size((batch_size * de_len, output_size + en_len)))

    def _init_dataset(self):
        test_path = os.path.dirname(os.path.realpath(__file__))
        src_path = os.path.join(test_path, 'data/src.txt')
        tgt_path = os.path.join(test_path, 'data/tgt.txt')
        dataset = Seq2SeqDataset.from_file(src_path, tgt_path, dynamic=True)
        dataset.build_vocab(100, 100)
        return dataset
