import os
import time
import shutil
import logging

import torch

from seq2seq.dataset.vocabulary import Vocabulary
from seq2seq.models.seq2seq import Seq2seq

logger = logging.getLogger(__name__)


class Checkpoint(object):
    """
    The Checkpoint class manages the saving and loading of a model during training. It allows training to be suspended
    and resumed at a later time (e.g. when running on a cluster using sequential jobs).

    To make a checkpoint, initialize a Checkpoint object with the following args; then call that object's save() method
    to write parameters to disk.

    Args:
        root_dir (str): main experiment directory
        model (seq2seq): seq2seq model being trained
        optimizer_state_dict (dict): stores the state of the optimizer
        epoch (int): current epoch (an epoch is a loop through the full training data)
        step (int): number of examples seen within the current epoch
        input_vocab (Vocabulary): vocabulary for the input language
        output_vocab (Vocabulary): vocabulary for the output language

    Attributes:
        CHECKPOINT_DIR_NAME (str): name of the checkpoint directory
        MODEL_DIR_NAME (str): name of the directory storing models
        INPUT_VOCAB_FILE (str): name of the input vocab file
        OUTPUT_VOCAB_FILE (str): name of the output vocab file
    """

    CHECKPOINT_DIR_NAME = 'checkpoints'
    MODEL_DIR_NAME = 'model_checkpoint'
    INPUT_VOCAB_FILE = 'input_vocab'
    OUTPUT_VOCAB_FILE = 'output_vocab'

    def __init__(self, root_dir, model, optimizer_state_dict, epoch, step, input_vocab,
                 output_vocab):
        self.root_dir = root_dir
        self.model = model
        self.optimizer_state_dict = optimizer_state_dict
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.epoch = epoch
        self.step = step

    def save(self):
        """
        Saves the current model and related training parameters into a subdirectory of the checkpoint directory.
        The name of the subdirectory is the current local time in Y_M_D_H_M_S format.

        Returns:
             str: path to the saved checkpoint subdirectory
        """
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        path = os.path.join(self.root_dir, self.CHECKPOINT_DIR_NAME, date_time)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'optimizer': self.optimizer_state_dict
        }, os.path.join(path, self.MODEL_DIR_NAME))
        self.model.save(path)

        if not os.path.isfile(os.path.join(path, self.INPUT_VOCAB_FILE)):
            self.input_vocab.save(os.path.join(path, self.INPUT_VOCAB_FILE))
        if not os.path.isfile(os.path.join(path, self.OUTPUT_VOCAB_FILE)):
            self.output_vocab.save(os.path.join(path, self.OUTPUT_VOCAB_FILE))

        return path

    @classmethod
    def load(cls, path):
        """
        Loads a Checkpoint object that was previously saved to disk.

        Args:
            path (str): path to the checkpoint subdirectory

        Returns:
            checkpoint (Checkpoint): checkpoint object with fields copied from those stored on disk
        """
        logger.info("Loading checkpoints from %s", path)
        resume_checkpoint = torch.load(os.path.join(path, cls.MODEL_DIR_NAME))
        # TODO: Check IBM repo for the proper fix.
        model = Seq2seq.load(path)
        input_vocab = Vocabulary.load(os.path.join(path, cls.INPUT_VOCAB_FILE))
        output_vocab = Vocabulary.load(os.path.join(path, cls.OUTPUT_VOCAB_FILE))
        return Checkpoint(
            root_dir=path,
            model=model,
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            optimizer_state_dict=resume_checkpoint['optimizer'],
            epoch=resume_checkpoint['epoch'],
            step=resume_checkpoint['step'])

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        """
        Given the path to an experiment directory, returns the path to the last saved checkpoint's subdirectory.

        Precondition: at least one checkpoint has been made (i.e., latest checkpoint subdirectory exists).

        Args:
            experiment_path (str): path to the experiment directory

        Returns:
             str: path to the last saved checkpoint's subdirectory
        """
        # get all times
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])
