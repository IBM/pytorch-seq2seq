from __future__ import division
import logging
import os
import random

import torch
import torchtext
from torch import optim
from tqdm import tqdm

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.util import Checkpoint

logger = logging.getLogger(__name__)


class SupervisedTrainer(object):
    """The SupervisedTrainer class helps in setting up a training framework
    in a supervised setting.

    Args:
        experiment_directory (optional, str): directory to store experiments in
        loss (seq2seq.loss.loss.Loss, optional): loss for training
        batch_size (int, optional): batch size for experiment
        checkpoint_every (int, optional): number of batches to checkpoint after
    """
    def __init__(self, experiment_directory='./experiment', loss=None, batch_size=64,
                random_seed=None, checkpoint_every=100, print_every=100):
        if loss is None:
            loss = NLLLoss()
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.batch_size = batch_size
        self.experiment_directory = experiment_directory

        if not os.path.exists(self.experiment_directory):
            os.makedirs(self.experiment_directory)

    def train(self, model, data, n_epochs=5, resume=False,
            dev_data=None, optimizer=None, teacher_forcing_ratio=0):
        """Train a given model.

        Args:
            model (seq2seq.models): model to run training on. If resume=True,
                it will be overwritten by the model loaded from the latest
                checkpoint
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            n_epochs (int): number of epochs to run
            resume(bool): resume training with the latest checkpoint
            dev_data (seq2seq.dataset.dataset.Dataset): dev Dataset
            optimizer (seq2seq.optim.Optimizer): optimizer for training
            teacher_forcing_ratio (float): teaching forcing ratio
        Returns:
            model (seq2seq.models): trained model.
        """
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(
                self.experiment_directory)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A work-around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(
                model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(
                    optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        logger.info('Optimizer: %s, Scheduler: %s',
                    self.optimizer.optimizer, self.optimizer.scheduler)

        self._train_epochs(data, model, n_epochs, 
                            start_epoch, step, dev_data=dev_data, 
                            teacher_forcing_ratio=teacher_forcing_ratio)
        return model

    def _train_epochs(self, data, model, n_epochs, start_epoch, 
                    start_step, dev_data=None, teacher_forcing_ratio=0):
        print_loss_total = epoch_loss_total = 0
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        batch_iterator = torchtext.data.BucketIterator(
            dataset=data,
            batch_size=self.batch_size,
            sort=False,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device,
            repeat=False,
        )

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs + 1):
            logger.debug('Epoch: %d, Step: %d', epoch, step)

            batch_generator = iter(batch_iterator)
            # Consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train()
            progress_bar = tqdm(
                batch_generator,
                total=steps_per_epoch,
                desc='Train {}: '.format(self.loss.name),
            )
            for batch in progress_bar:
                step += 1
                step_elapsed += 1

                loss = self._train_batch(
                    batch,
                    model,
                    teacher_forcing_ratio,
                    data,
                )
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 \
                   and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    progress_bar.set_description('Train {}: {:.4f}'.format(
                        self.loss.name,
                        print_loss_avg,
                    ))

                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(
                        model=model,
                        optimizer=self.optimizer,
                        epoch=epoch, step=step,
                        input_vocab=data.fields[seq2seq.src_field_name].vocab,
                        output_vocab=data.fields[seq2seq.tgt_field_name].vocab,
                    ).save(self.experiment_directory)

            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / min(
                steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = 'Finished epoch {:d}: Train {}: {:.4f}'.format(
                epoch, self.loss.name, epoch_loss_avg)
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                self.optimizer.update(dev_loss, epoch)
                log_msg += ', Dev {}: {:.4f}, Accuracy: {:.4f}'.format(
                    self.loss.name, dev_loss, accuracy)
                model.train()
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            logger.info(log_msg)

    def _train_batch(self, batch, model, teacher_forcing_ratio, dataset):
        # Forward propagation
        output, _, _ = model(
            batch,
            dataset=dataset,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        # Get loss
        self.loss.reset()
        self.loss.eval_batch(output, batch)

        # Backward propagation
        model.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        return self.loss.get_loss()
