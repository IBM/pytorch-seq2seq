import logging
import os
import random
import time

import torch
import torchtext
from torch import optim

from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.util.custom_time import pretty_interval
from seq2seq.util.checkpoint import Checkpoint

class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of epochs to checkpoint after, (default: 100)
        optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
            (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
    """
    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, print_every=100,
                 optimizer=Optimizer(optim.Adam, max_grad_norm=5)):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = optimizer
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio):
        loss = self.loss
        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable,
                                                       teacher_forcing_ratio=teacher_forcing_ratio)
        # Get loss
        loss.reset()
        lengths = other['length']
        for batch in range(target_variable.size(0)):
            # Batch wise loss
            batch_target = target_variable[batch, 1:]
            batch_len = min(lengths[batch], target_variable.size(1) - 1)
            # Crop output and target to batch length
            batch_output = torch.stack([output[batch] for output in decoder_outputs[:batch_len]])
            batch_target = batch_target[:batch_len]
            # Evaluate loss
            loss.eval_batch(batch_output, batch_target)
        # Backward propagation
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, n_epochs, batch_size, resume, dev_data=None, teacher_forcing_ratio=0):
        start = time.time()
        print_loss_total = 0  # Reset every print_every
        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort_key=lambda x: -len(x.src),
            device=device, repeat=False)
        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer.set_parameters(model.parameters())
            self.optimizer.load_state_dict(resume_checkpoint.optimizer_state_dict)
            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            self.optimizer.set_parameters(model.parameters())

        for epoch in range(start_epoch, n_epochs + 1):

            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                for _ in batch_iterator: break

            model.train(True)
            for batch in batch_iterator:
                step += 1

                input_variables, input_lengths = batch.src
                target_variables = batch.trg

                loss = self._train_batch(input_variables, input_lengths.tolist(), target_variables, model, teacher_forcing_ratio)

                # Record average loss
                print_loss_total += loss

                if step % self.print_every == 0:
                    print_loss_avg = print_loss_total / (self.print_every)
                    print_loss_total = 0
                    log_msg = 'Time elapsed: %s, Progress: %d%%, Train %s: %.4f' % (
                        pretty_interval(start),
                        float(step) / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    self.logger.info(log_msg)

                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(model=model,
                               optimizer_state_dict=self.optimizer.state_dict(),
                               epoch=epoch, step=step,
                               input_vocab=data.fields['src'].vocab,
                               output_vocab=data.fields['trg'].vocab).save(self.expt_dir)

            log_msg = "Finished epoch {0}".format(epoch)
            if dev_data is not None:
                dev_loss = self.evaluator.evaluate(model, dev_data)
                self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev %s: %.4f" % (self.loss.name, dev_loss)
                model.train(mode=True)
            self.logger.info(log_msg)

    def train(self, model, data, num_epochs=5, resume=False, dev_data=None, teacher_forcing_ratio=0):
        """ Run training for a given model.

         Args:
             model (seq2seq.models): model to run training on, if `resume=True`, it would be
                overwritten by the model loaded from the latest checkpoint.
             data (seq2seq.dataset.dataset.Dataset): dataset object to train on
             num_epochs (int, optional): number of epochs to run (default 5)
             resume(bool, optional): resume training with the latest checkpoint, (default False)
             dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
             teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)

        """
        self._train_epoches(data, model, num_epochs, self.batch_size,
                            resume=resume, dev_data=dev_data, teacher_forcing_ratio=teacher_forcing_ratio)
