import unittest
import os
import shutil

import mock
from mock import ANY

from seq2seq.util.checkpoint import Checkpoint


class TestCheckpoint(unittest.TestCase):

    EXP_DIR = "test_experiment"

    def tearDown(self):
        path = self._get_experiment_dir()
        if os.path.exists(path):
            shutil.rmtree(path)

    def test_path_error(self):
        ckpt = Checkpoint(None, None, None, None, None, None)
        self.assertRaises(LookupError, lambda: ckpt.path)

    @mock.patch('seq2seq.util.checkpoint.os.listdir')
    def test_get_latest_checkpoint(self, mock_listdir):
        mock_listdir.return_value = ['2017_05_22_09_47_26',
                                     '2017_05_22_09_47_31',
                                     '2017_05_23_10_47_29']
        latest_checkpoint = Checkpoint.get_latest_checkpoint(self.EXP_DIR)
        self.assertEquals(latest_checkpoint,
                          os.path.join(self.EXP_DIR,
                                       'checkpoints/2017_05_23_10_47_29'))

    @mock.patch('seq2seq.util.checkpoint.torch')
    @mock.patch('seq2seq.util.checkpoint.dill')
    @mock.patch('seq2seq.util.checkpoint.open')
    def test_save_checkpoint_calls_torch_save(self, mock_open, mock_dill, mock_torch):
        epoch = 5
        step = 10
        optim = mock.Mock()
        state_dict = {'epoch': epoch, 'step': step, 'optimizer': optim}

        mock_model = mock.Mock()
        mock_vocab = mock.Mock()
        mock_open.return_value = mock.MagicMock()

        chk_point = Checkpoint(model=mock_model, optimizer=optim,
                               epoch=epoch, step=step,
                               input_vocab=mock_vocab, output_vocab=mock_vocab)

        path = chk_point.save(self._get_experiment_dir())

        self.assertEquals(2, mock_torch.save.call_count)
        mock_torch.save.assert_any_call(state_dict,
                                        os.path.join(chk_point.path, Checkpoint.TRAINER_STATE_NAME))
        mock_torch.save.assert_any_call(mock_model,
                                        os.path.join(chk_point.path, Checkpoint.MODEL_NAME))
        self.assertEquals(2, mock_open.call_count)
        mock_open.assert_any_call(os.path.join(path, Checkpoint.INPUT_VOCAB_FILE), ANY)
        mock_open.assert_any_call(os.path.join(path, Checkpoint.OUTPUT_VOCAB_FILE), ANY)
        self.assertEquals(2, mock_dill.dump.call_count)
        mock_dill.dump.assert_any_call(mock_vocab,
                                       mock_open.return_value.__enter__.return_value)

    @mock.patch('seq2seq.util.checkpoint.torch')
    @mock.patch('seq2seq.util.checkpoint.dill')
    @mock.patch('seq2seq.util.checkpoint.open')
    def test_load(self, mock_open, mock_dill, mock_torch):
        dummy_vocabulary = mock.Mock()
        mock_optimizer = mock.Mock()
        torch_dict = {"optimizer": mock_optimizer, "epoch": 5, "step": 10}
        mock_open.return_value = mock.MagicMock()
        mock_torch.load.side_effect = [torch_dict, mock.MagicMock()]
        mock_dill.load.return_value = dummy_vocabulary

        loaded_chk_point = Checkpoint.load("mock_checkpoint_path")

        mock_torch.load.assert_any_call(
            os.path.join('mock_checkpoint_path', Checkpoint.TRAINER_STATE_NAME))
        mock_torch.load.assert_any_call(
            os.path.join("mock_checkpoint_path", Checkpoint.MODEL_NAME))

        self.assertEquals(loaded_chk_point.epoch, torch_dict['epoch'])
        self.assertEquals(loaded_chk_point.optimizer, torch_dict['optimizer'])
        self.assertEquals(loaded_chk_point.step, torch_dict['step'])
        self.assertEquals(loaded_chk_point.input_vocab, dummy_vocabulary)
        self.assertEquals(loaded_chk_point.output_vocab, dummy_vocabulary)

    def _get_experiment_dir(self):
        root_dir = os.path.dirname(os.path.realpath(__file__))
        experiment_dir = os.path.join(root_dir, self.EXP_DIR)
        return experiment_dir


if __name__ == '__main__':
    unittest.main()
