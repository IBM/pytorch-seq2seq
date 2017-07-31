import os
import argparse
import logging
from logging.config import fileConfig

import torch

from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.dataset import Dataset
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


opt = parser.parse_args()
logger.info(opt)

if opt.load_checkpoint is not None:
    logger.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    dataset = Dataset.from_file(opt.train_path, src_max_len=50, tgt_max_len=50)
    input_vocab = dataset.input_vocab
    output_vocab = dataset.output_vocab

    dev_set = Dataset.from_file(opt.dev_path, src_max_len=50, tgt_max_len=50,
                    src_vocab=input_vocab,
                    tgt_vocab=output_vocab)

    # Prepare loss
    weight = torch.ones(output_vocab.get_vocab_size())
    mask = output_vocab.MASK_token_id
    loss = Perplexity(weight, mask)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    if not opt.resume:
        # Initialize model
        hidden_size=128
        encoder = EncoderRNN(input_vocab, dataset.src_max_len, hidden_size)
        decoder = DecoderRNN(output_vocab, dataset.tgt_max_len, hidden_size,
                            dropout_p=0.2, use_attention=True)
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=32,
                        checkpoint_every=50,
                        print_every=10, expt_dir=opt.expt_dir)
    t.train(seq2seq, dataset, num_epochs=4, dev_data=dev_set, resume=opt.resume)

predictor = Predictor(seq2seq, input_vocab, output_vocab)

while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.split()
    print(predictor.predict(seq))
