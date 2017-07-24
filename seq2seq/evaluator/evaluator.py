from __future__ import print_function

import torch

from seq2seq.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.
    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()
        loss = self.loss
        loss.reset()

        for batch in data.make_batches(self.batch_size):
            input_variables = batch[0]
            target_variables = batch[1]

            decoder_outputs, decoder_hidden, other = model(input_variables, target_variables, volatile=True)

            # Evaluation
            targets = other['inputs']
            lengths = other['length']
            for b in range(len(targets)):
                # Batch wise loss
                batch_target = targets[b]
                batch_len = lengths[b]
                # Crop output and target to batch length
                batch_output = torch.stack([output[b] for output in decoder_outputs[:batch_len]])
                batch_target = batch_target[:batch_len]
                # Evaluate loss
                loss.eval_batch(batch_output, batch_target)

        model.train(True)

        return loss.get_loss()
