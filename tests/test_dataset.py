import os
import unittest

import torchtext

import seq2seq
from seq2seq.data import Seq2SeqDataset

class TestDataset(unittest.TestCase):

    test_path = os.path.dirname(os.path.realpath(__file__))
    src_path = os.path.join(test_path, 'data/src.txt')
    tgt_path = os.path.join(test_path, 'data/tgt.txt')

    def test_init_ONLY_SRC(self):
        dataset = Seq2SeqDataset.from_file(self.src_path, dynamic=False)
        self.assertEqual(len(dataset.fields), 2)
        self.assertEqual(len(dataset), 100)
        self.assertTrue(hasattr(dataset.examples[0], seq2seq.src_field_name))

    def test_init_SRC_AND_TGT(self):
        dataset = Seq2SeqDataset.from_file(self.src_path, self.tgt_path, dynamic=False)
        self.assertEqual(len(dataset.fields), 3)
        self.assertEqual(len(dataset), 100)
        ex = dataset.examples[0]
        self.assertTrue(len(getattr(ex, seq2seq.tgt_field_name)) > 2)

    def test_indices(self):
        dataset = Seq2SeqDataset.from_file(self.src_path, self.tgt_path, dynamic=False)
        dataset.build_vocab(1000, 1000)
        batch_size = 25

        generator = torchtext.data.BucketIterator(dataset, batch_size, device=-1)
        batch = next(generator.__iter__())
        self.assertTrue(hasattr(batch, 'index'))

    def test_init_FROM_LIST(self):
        src_list = [['1','2','3'], ['4','5','6','7']]
        dataset = Seq2SeqDataset.from_list(src_list, dynamic=False)

        self.assertEqual(len(dataset), 2)

        tmp_file = open('temp', 'w')
        for seq in src_list:
            tmp_file.write(' '.join(seq) + "\n")
        tmp_file.close()
        from_file = Seq2SeqDataset.from_file('temp', dynamic=False)

        self.assertEqual(len(dataset.examples), len(from_file.examples))
        for l, f in zip(dataset.examples, from_file.examples):
            self.assertEqual(l.src, f.src)
        os.remove('temp')

    def test_dynamic(self):
        dataset = Seq2SeqDataset.from_file(self.src_path, self.tgt_path, dynamic=True)
        self.assertTrue('src_index' in dataset.fields)
        for i, ex in enumerate(dataset.examples):
            idx = ex.index
            self.assertEqual(i, idx)
            src_vocab = dataset.dynamic_vocab[i]
            for tok, tok_id in zip(ex.src, ex.src_index):
                self.assertEqual(src_vocab.stoi[tok], tok_id)
