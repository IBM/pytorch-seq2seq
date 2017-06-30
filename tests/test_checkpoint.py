import unittest
import os


import mock
from mock import ANY

from seq2seq.util.checkpoint import Checkpoint
import time


class TestCheckpoint(unittest.TestCase):

    EXP_DIR = "experiment"

    @mock.patch('seq2seq.util.checkpoint.os.listdir')
    def test_get_latest_checkpoint(self,mock_listdir):
        mock_listdir.return_value = ['2017_05_22_09_47_26', '2017_05_22_09_47_31', '2017_05_23_10_47_29']
        latest_checkpoint = Checkpoint.get_latest_checkpoint(self.EXP_DIR)
        self.assertTrue(latest_checkpoint == os.path.join(self.EXP_DIR, 'checkpoints/2017_05_23_10_47_29'))

    @mock.patch('seq2seq.util.checkpoint.torch')
    def test_save_checkpoint_calls_torch_save_with_right_arguments(self,mock_torch):
        mock_model = mock.Mock()
        opt_state_dict = {"key2":"val2"}

        epoch = 5
        step = 10

        root_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.join(root_dir, self.EXP_DIR)

        chk_point = Checkpoint(root_dir=root_dir, model=mock_model, optimizer_state_dict=opt_state_dict, epoch=epoch, step=step,
                               input_vocab=mock.Mock(), output_vocab=mock.Mock())
        path = chk_point.save()

        mock_model.save.assert_called_once_with(ANY)
        state_dict = {'epoch': epoch, 'step': step, 'optimizer': opt_state_dict}

        mock_torch.save.assert_called_once_with(state_dict, ANY)
        os.system("rm -rf " + path)

    @mock.patch('seq2seq.util.checkpoint.os.path.isfile')
    def test_save_checkpoint_saves_vocab_if_not_exist(self, mock_os_path_exists):
        mock_model = mock.Mock()
        model_dict = {"key1": "val1"}
        mock_model.state_dict.return_value = model_dict

        opt_dict = {"key2":"val2"}

        epoch = 5
        step = 10

        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        root_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.join(root_dir, self.EXP_DIR)

        mock_os_path_exists.return_value = False
        input_vocab = mock.Mock()
        output_vocab = mock.Mock()

        chk_point = Checkpoint(root_dir=root_dir, model=mock_model, optimizer_state_dict=opt_dict, epoch=epoch, step=step,
                               input_vocab=input_vocab, output_vocab=output_vocab)
        path = chk_point.save()

        input_vocab.save.assert_called_once_with(os.path.join(root_dir, Checkpoint.CHECKPOINT_DIR_NAME,
                                                              date_time, "input_vocab"))
        output_vocab.save.assert_called_once_with(os.path.join(root_dir, Checkpoint.CHECKPOINT_DIR_NAME,
                                                               date_time, "output_vocab"))

        os.system("rm -rf " + path )


    @mock.patch('seq2seq.util.checkpoint.torch')
    @mock.patch('seq2seq.util.checkpoint.Vocabulary')
    def test_load(self, mock_vocabulary, mock_torch):
        dummy_vocabulary = mock.Mock()
        mock_optimizer_state_dict = mock.Mock()
        torch_dict = {"optimizer": mock_optimizer_state_dict, "epoch": 5, "step": 10}
        mock_torch.load.return_value = torch_dict

        mock_vocabulary.load.return_value = dummy_vocabulary
        loaded_chk_point = Checkpoint.load("mock_checkpoint_path")


        mock_torch.load.assert_any_call(os.path.join('mock_checkpoint_path','model_checkpoint'))
        mock_torch.load.assert_any_call("mock_checkpoint_path")

        self.assertEquals(loaded_chk_point.epoch, torch_dict['epoch'])
        self.assertEquals(loaded_chk_point.optimizer_state_dict, torch_dict['optimizer'])
        self.assertEquals(loaded_chk_point.step, torch_dict['step'])

if __name__ == '__main__':
    unittest.main()
