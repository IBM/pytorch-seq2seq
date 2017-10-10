import os
import unittest

import torchtext

import seq2seq
from seq2seq.dataset import Seq2SeqDataset

class TestDataset(unittest.TestCase):

    test_path = os.path.dirname(os.path.realpath(__file__))
    src_path = os.path.join(test_path, 'data/src.txt')
    tgt_path = os.path.join(test_path, 'data/tgt.txt')

    def test_init_ONLY_SRC(self):
        dataset = Seq2SeqDataset(self.src_path)
        self.assertEqual(len(dataset.fields), 2)
        self.assertEqual(len(dataset), 100)
        self.assertTrue(hasattr(dataset.examples[0], seq2seq.src_field_name))

    def test_init_SRC_AND_TGT(self):
        dataset = Seq2SeqDataset(self.src_path, self.tgt_path)
        self.assertEqual(len(dataset.fields), 3)
        self.assertEqual(len(dataset), 100)
        ex = dataset.examples[0]
        self.assertTrue(len(getattr(ex, seq2seq.tgt_field_name)) > 2)

    def test_indices(self):
        dataset = Seq2SeqDataset(self.src_path, self.tgt_path)
        dataset.build_vocab(1000, 1000)
        batch_size = 25

        generator = torchtext.data.BucketIterator(dataset, batch_size, device=-1)
        batch = next(generator.__iter__())
        self.assertTrue(hasattr(batch, 'index'))
