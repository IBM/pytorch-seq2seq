import unittest
import os

import mock
import torchtext

from seq2seq.dataset import SourceField, TargetField
from seq2seq.trainer import SupervisedTrainer

class TestSupervisedTrainer(unittest.TestCase):

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

    @mock.patch('seq2seq.trainer.SupervisedTrainer._train_batch', return_value=0)
    @mock.patch('seq2seq.util.checkpoint.Checkpoint.save')
    def test_batch_num_when_resuming(self, mock_checkpoint, mock_func):
        mock_model = mock.Mock()
        mock_optim = mock.Mock()

        trainer = SupervisedTrainer(batch_size=16)
        trainer.optimizer = mock_optim
        n_epoches = 1
        start_epoch = 1
        steps_per_epoch = 7
        step = 3
        trainer._train_epoches(self.dataset, mock_model, n_epoches, start_epoch, step)

        self.assertEqual(steps_per_epoch - step, mock_func.call_count)

    @mock.patch('seq2seq.trainer.SupervisedTrainer._train_batch', return_value=0)
    @mock.patch('seq2seq.util.checkpoint.Checkpoint.save')
    def test_resume_from_multiple_of_epoches(self, mock_checkpoint, mock_func):
        mock_model = mock.Mock()
        mock_optim = mock.Mock()

        trainer = SupervisedTrainer(batch_size=16)
        trainer.optimizer = mock_optim
        n_epoches = 1
        start_epoch = 1
        step = 7
        trainer._train_epoches(self.dataset, mock_model, n_epoches, start_epoch, step)

if __name__ == '__main__':
    unittest.main()
