from __future__ import print_function

import torch
import torchtext

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

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort_key=lambda batch: -len(batch.src),
            device=device, train=False)

        for batch in batch_iterator:
            input_variables, input_lengths  = batch.src
            target_variables = batch.trg

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Evaluation
            lengths = other['length']
            for b in range(target_variables.size(0)):
                # Batch wise loss
                batch_target = target_variables[b, 1:]
                batch_len = min(lengths[b], target_variables.size(1) - 1)
                # Crop output and target to batch length
                batch_output = torch.stack([output[b] for output in decoder_outputs[:batch_len]])
                batch_target = batch_target[:batch_len]
                # Evaluate loss
                loss.eval_batch(batch_output, batch_target)

        return loss.get_loss()
