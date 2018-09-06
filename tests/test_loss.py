import math
import random
import unittest

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext

from seq2seq.loss.loss import Loss
from seq2seq.loss import NLLLoss, Perplexity
from seq2seq.data import Seq2SeqDataset

class TestLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        num_class = 5
        cls.num_class = 5
        batch_size = 5
        length = 7
        cls.outputs = [F.softmax(Variable(torch.randn(batch_size, num_class)), dim=-1) for _ in range(length)]
        targets = [random.randint(0, num_class - 1) for _ in range(batch_size * (length + 1))]
        targets_list = [str(x) for x in targets]
        sources = ['0'] * len(targets)
        dataset = Seq2SeqDataset.from_list(sources, targets_list)
        dataset.build_vocab(5, 5)
        cls.targets = Variable(torch.LongTensor(targets)).view(batch_size, length + 1)
        cls.batch = torchtext.data.Batch.fromvars(dataset, batch_size, tgt=cls.targets)

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
        num_batch = 10
        loss = NLLLoss()
        pytorch_loss = 0
        pytorch_criterion = torch.nn.NLLLoss()
        for _ in range(num_batch):
            for step, output in enumerate(self.outputs):
                pytorch_loss += pytorch_criterion(output, self.targets[:, step + 1])
            loss.eval_batch(self.outputs, self.batch)

        loss_val = loss.get_loss()
        pytorch_loss /= (num_batch * len(self.outputs))

        self.assertAlmostEqual(loss_val, pytorch_loss.item())

    def test_nllloss_WITH_OUT_SIZE_AVERAGE(self):
        num_repeat = 10
        loss = NLLLoss(reduction='sum')
        pytorch_loss = 0
        pytorch_criterion = torch.nn.NLLLoss(reduction='sum')
        for _ in range(num_repeat):
            for step, output in enumerate(self.outputs):
                pytorch_loss += pytorch_criterion(output, self.targets[:, step + 1])
            loss.eval_batch(self.outputs, self.batch)

        loss_val = loss.get_loss()

        self.assertAlmostEqual(loss_val, pytorch_loss.item())

    def test_perplexity_init(self):
        loss = Perplexity()
        self.assertEqual(loss.name, Perplexity._NAME)

    def test_perplexity(self):
        nll = NLLLoss()
        ppl = Perplexity()
        nll.eval_batch(self.outputs, self.batch)
        ppl.eval_batch(self.outputs, self.batch)

        nll_loss = nll.get_loss()
        ppl_loss = ppl.get_loss()

        self.assertAlmostEqual(ppl_loss, math.exp(nll_loss))