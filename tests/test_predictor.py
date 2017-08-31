import os
import unittest

import torchtext

from seq2seq.evaluator import Predictor
from seq2seq.dataset import SourceField, TargetField
from seq2seq.models import Seq2seq, EncoderRNN, DecoderRNN

class TestPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        test_path = os.path.dirname(os.path.realpath(__file__))
        src = SourceField()
        trg = TargetField()
        dataset = torchtext.data.TabularDataset(
            path=os.path.join(test_path, 'data/eng-fra.txt'), format='tsv',
            fields=[('src', src), ('trg', trg)],
        )
        src.build_vocab(dataset)
        trg.build_vocab(dataset)

        encoder = EncoderRNN(len(src.vocab), 10, 10, rnn_cell='lstm')
        decoder = DecoderRNN(len(trg.vocab), 10, 10, trg.sos_id, trg.eos_id, rnn_cell='lstm')
        seq2seq = Seq2seq(encoder, decoder)
        self.predictor = Predictor(seq2seq, src.vocab, trg.vocab)

    def test_predict(self):
        src_seq = ["I", "am", "fat"]
        tgt_seq = self.predictor.predict(src_seq)
        for tok in tgt_seq:
            self.assertTrue(tok in self.predictor.tgt_vocab.stoi)
