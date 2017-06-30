import math
import random
import unittest

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from seq2seq.loss.loss import Loss
from seq2seq.loss import NLLLoss, Perplexity

class TestLoss(unittest.TestCase):

    def test_loss_init(self):
        name = "name"
        loss = Loss(name, torch.nn.NLLLoss())
        self.assertEqual(loss.name, name)

    def test_loss_init_WITH_NON_LOSS(self):
        self.assertRaises(ValueError, lambda: Loss("name", "loss"))

    def test_loss_backward_WITH_NO_LOSS(self):
        loss = Loss("name", torch.nn.NLLLoss())
        self.assertRaises(ValueError, lambda: loss.backward())

    def test_nllloss_init(self):
        loss = NLLLoss()
        self.assertEqual(loss.name, NLLLoss._NAME)
        self.assertTrue(type(loss.criterion) is torch.nn.NLLLoss)

    def test_nllloss_init_WITH_MASK_BUT_NO_WEIGHT(self):
        mask = 1
        self.assertRaises(ValueError, lambda: NLLLoss(mask=mask))

    def test_nllloss(self):
        num_class = 5
        num_batch = 10
        batch_size = 5

        outputs = [F.softmax(Variable(torch.randn(batch_size, num_class)))
                   for _ in range(num_batch)]
        targets = [Variable(torch.LongTensor([random.randint(0, num_class - 1)
                                              for _ in range(batch_size)]))
                   for _ in range(num_batch)]

        loss = NLLLoss()
        pytorch_loss = 0
        pytorch_criterion = torch.nn.NLLLoss()
        for output, target in zip(outputs, targets):
            loss.eval_batch(output, target)
            pytorch_loss += pytorch_criterion(output, target)

        loss_val = loss.get_loss()
        pytorch_loss /= num_batch

        self.assertAlmostEqual(loss_val, pytorch_loss.data[0])

    def test_perplexity_init(self):
        loss = Perplexity()
        self.assertEqual(loss.name, Perplexity._NAME)

    def test_perplexity(self):
        num_class = 5
        num_batch = 10
        batch_size = 5

        outputs = [F.softmax(Variable(torch.randn(batch_size, num_class)))
                   for _ in range(num_batch)]
        targets = [Variable(torch.LongTensor([random.randint(0, num_class - 1)
                                              for _ in range(batch_size)]))
                   for _ in range(num_batch)]

        nll = NLLLoss()
        ppl = Perplexity()
        for output, target in zip(outputs, targets):
            nll.eval_batch(output, target)
            ppl.eval_batch(output, target)

        nll_loss = nll.get_loss()
        ppl_loss = ppl.get_loss()

        self.assertAlmostEqual(ppl_loss, math.exp(nll_loss))
