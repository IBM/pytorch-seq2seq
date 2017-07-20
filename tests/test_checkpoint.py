import unittest
import os
import shutil
import mock

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
    def test_save_checkpoint_calls_torch_save(self, mock_torch):
        epoch = 5
        step = 10
        opt_state_dict = {"key2": "val2"}
        state_dict = {'epoch': epoch, 'step': step, 'optimizer': opt_state_dict}

        mock_model = mock.Mock()

        chk_point = Checkpoint(model=mock_model, optimizer_state_dict=opt_state_dict,
                               epoch=epoch, step=step,
                               input_vocab=mock.Mock(), output_vocab=mock.Mock())
        chk_point.save(self._get_experiment_dir())

        self.assertEquals(2, mock_torch.save.call_count)
        mock_torch.save.assert_any_call(state_dict,
                                        os.path.join(chk_point.path, Checkpoint.TRAINER_STATE_NAME))
        mock_torch.save.assert_any_call(mock_model,
                                        os.path.join(chk_point.path, Checkpoint.MODEL_NAME))

    @mock.patch('seq2seq.util.checkpoint.torch')
    @mock.patch('seq2seq.util.checkpoint.os.path.isfile', return_value=False)
    def test_save_checkpoint_saves_vocab_if_not_exist(self, mock_torch, mock_os_path_isfile):
        epoch = 5
        step = 10
        model_dict = {"key1": "val1"}
        opt_dict = {"key2": "val2"}

        mock_model = mock.Mock()
        mock_model.state_dict.return_value = model_dict

        input_vocab = mock.Mock()
        output_vocab = mock.Mock()

        chk_point = Checkpoint(model=mock_model, optimizer_state_dict=opt_dict, epoch=epoch, step=step,
                               input_vocab=input_vocab, output_vocab=output_vocab)
        chk_point.save(self._get_experiment_dir())

        input_vocab.save.assert_called_once_with(os.path.join(chk_point.path, "input_vocab"))
        output_vocab.save.assert_called_once_with(os.path.join(chk_point.path, "output_vocab"))

    @mock.patch('seq2seq.util.checkpoint.torch')
    @mock.patch('seq2seq.util.checkpoint.Vocabulary')
    def test_load(self, mock_vocabulary, mock_torch):
        dummy_vocabulary = mock.Mock()
        mock_optimizer_state_dict = mock.Mock()
        torch_dict = {"optimizer": mock_optimizer_state_dict, "epoch": 5, "step": 10}
        mock_torch.load.return_value = torch_dict
        mock_vocabulary.load.return_value = dummy_vocabulary

        loaded_chk_point = Checkpoint.load("mock_checkpoint_path")

        mock_torch.load.assert_any_call(
            os.path.join('mock_checkpoint_path', Checkpoint.TRAINER_STATE_NAME))
        mock_torch.load.assert_any_call(
            os.path.join("mock_checkpoint_path", Checkpoint.MODEL_NAME))

        self.assertEquals(loaded_chk_point.epoch, torch_dict['epoch'])
        self.assertEquals(loaded_chk_point.optimizer_state_dict,
                          torch_dict['optimizer'])
        self.assertEquals(loaded_chk_point.step, torch_dict['step'])

    def _get_experiment_dir(self):
        root_dir = os.path.dirname(os.path.realpath(__file__))
        experiment_dir = os.path.join(root_dir, self.EXP_DIR)
        return experiment_dir


if __name__ == '__main__':
    unittest.main()
