import os
import unittest

from mock import MagicMock, patch, call, ANY
import torch

from seq2seq.evaluator import Evaluator
from seq2seq.dataset import Dataset
from seq2seq.models import Seq2seq, EncoderRNN, DecoderRNN

class TestPredictor(unittest.TestCase):

    def setUp(self):
        self.test_wd = os.getcwd()
        self.dataset = Dataset(path=os.path.join(self.test_wd, 'tests/data/eng-fra.txt'),
                               src_max_len=50, tgt_max_len=50, src_max_vocab=50000, tgt_max_vocab=50000)
        self.encoder = EncoderRNN(self.dataset.input_vocab, max_len=10, hidden_size=10, rnn_cell='lstm')
        self.decoder = DecoderRNN(self.dataset.output_vocab, max_len=10, hidden_size=10, rnn_cell='lstm')
        self.seq2seq = Seq2seq(self.encoder, self.decoder)
        if torch.cuda.is_available():
            self.seq2seq.cuda()
        self.mock_seq2seq = Seq2seq(self.encoder, self.decoder)

        for param in self.seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    @patch.object(Seq2seq, 'train')
    @patch.object(Seq2seq, '__call__', return_value=(None, None, dict(inputs=[], length=None)))
    @patch.object(Seq2seq, 'eval')
    def test_set_eval_mode(self, mock_eval, mock_call, mock_train):
        """ Make sure that evaluation is done in evaluation mode. """
        mock_mgr = MagicMock()
        mock_mgr.attach_mock(mock_eval, 'eval')
        mock_mgr.attach_mock(mock_call, 'call')
        mock_mgr.attach_mock(mock_train, 'train')

        evaluator = Evaluator()
        evaluator.evaluate(self.seq2seq, self.dataset)

        expected_calls = [call.eval()] + \
            self.dataset.num_batches(evaluator.batch_size) * [call.call(ANY, ANY, volatile=ANY)] + \
            [call.train(True)]
        self.assertEquals(expected_calls, mock_mgr.mock_calls)
