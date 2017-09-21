from __future__ import division
import os
import math
import unittest

from mock import MagicMock, patch, call, ANY
import torchtext

from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
from seq2seq.models import Seq2seq, EncoderRNN, DecoderRNN

class TestPredictor(unittest.TestCase):

    def setUp(self):
        test_path = os.path.dirname(os.path.realpath(__file__))
        src = SourceField()
        tgt = TargetField()
        self.dataset = torchtext.data.TabularDataset(
            path=os.path.join(test_path, 'data/eng-fra.txt'), format='tsv',
            fields=[('src', src), ('tgt', tgt)],
        )
        src.build_vocab(self.dataset)
        tgt.build_vocab(self.dataset)

        encoder = EncoderRNN(len(src.vocab), 10, 10, rnn_cell='lstm')
        decoder = DecoderRNN(len(tgt.vocab), 10, 10, tgt.sos_id, tgt.eos_id, rnn_cell='lstm')
        self.seq2seq = Seq2seq(encoder, decoder)

        for param in self.seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    @patch.object(Seq2seq, '__call__', return_value=([], None, dict(inputs=[], length=[10]*64, sequence=MagicMock())))
    @patch.object(Seq2seq, 'eval')
    def test_set_eval_mode(self, mock_eval, mock_call):
        """ Make sure that evaluation is done in evaluation mode. """
        mock_mgr = MagicMock()
        mock_mgr.attach_mock(mock_eval, 'eval')
        mock_mgr.attach_mock(mock_call, 'call')

        evaluator = Evaluator(batch_size=64)
        with patch('seq2seq.evaluator.evaluator.torch.stack', return_value=None), \
                patch('seq2seq.loss.NLLLoss.eval_batch', return_value=None):
            evaluator.evaluate(self.seq2seq, self.dataset)

        num_batches = int(math.ceil(len(self.dataset) / evaluator.batch_size))
        expected_calls = [call.eval()] + num_batches * [call.call(ANY, ANY, ANY)]
        self.assertEquals(expected_calls, mock_mgr.mock_calls)
