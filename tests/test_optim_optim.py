import unittest

import torch
import mock

from seq2seq.optim import Optimizer

class TestOptimizer(unittest.TestCase):

    def test_init(self):
        try:
            optimizer = Optimizer(torch.optim.SGD)
        except:
            self.fail("__init__ failed.")

        self.assertEquals(optimizer.max_grad_norm, 0)
        self.assertEquals(optimizer.lr_decay, 1)
        self.assertEquals(optimizer.decay_after_epoch, 0)

    def test_set_parameters(self):
        learning_rate = 1
        optim = Optimizer(torch.optim.SGD, lr=learning_rate)
        params = [torch.nn.Parameter(torch.randn(2,3,4))]
        optim.set_parameters(params)

        self.assertTrue(type(optim.optimizer) is torch.optim.SGD)
        self.assertEquals(optim.optimizer.param_groups[0]['lr'], learning_rate)

    def test_update(self):
        optim = Optimizer(torch.optim.SGD,
                          lr=1,
                          decay_after_epoch=5,
                          lr_decay=0.5)
        params = [torch.nn.Parameter(torch.randn(2,3,4))]
        optim.set_parameters(params)
        optim.update(0, 10)
        self.assertEquals(optim.optimizer.param_groups[0]['lr'], 0.5)

    @mock.patch("torch.nn.utils.clip_grad_norm")
    def test_step(self, mock_clip_grad_norm):
        optim = Optimizer(torch.optim.Adam,
                          max_grad_norm=5)
        params = [torch.nn.Parameter(torch.randn(2,3,4))]
        optim.set_parameters(params)
        optim.step()
        mock_clip_grad_norm.assert_called_once()

