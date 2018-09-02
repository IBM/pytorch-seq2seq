import os
import time
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import Seq2SeqDataset
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:

#     TRAIN_SRC=data/toy_reverse/train/src.txt
#     TRAIN_TGT=data/toy_reverse/train/tgt.txt
#     DEV_SRC=data/toy_reverse/dev/src.txt
#     DEV_TGT=data/toy_reverse/dev/tgt.txt
#
#     # training
#     python examples/sample.py  --train_src $TRAIN_SRC --train_tgt $TRAIN_TGT --dev_src $DEV_SRC --dev_tgt $DEV_TGT --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#     python examples/sample.py  --train_src $TRAIN_SRC --train_tgt $TRAIN_TGT --dev_src $DEV_SRC --dev_tgt $DEV_TGT --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#     python examples/sample.py  --train_src $TRAIN_SRC --train_tgt $TRAIN_TGT --dev_src $DEV_SRC --dev_tgt $DEV_TGT --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_src', action='store', help='Path to source train data')
parser.add_argument('--train_tgt', action='store', help='Path to target train data')
parser.add_argument('--dev_src', action='store', help='Path to source develop data')
parser.add_argument('--dev_tgt', action='store', help='Path to source develop data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    train = Seq2SeqDataset.from_file(opt.train_src, opt.train_tgt)
    train.build_vocab(50000, 50000)
    dev = Seq2SeqDataset.from_file(opt.dev_src, opt.dev_tgt, share_fields_from=train)
    input_vocab = train.src_field.vocab
    output_vocab = train.tgt_field.vocab

    # Prepare loss
    weight = torch.ones(len(output_vocab))
    pad = output_vocab.stoi[train.tgt_field.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size=128
        bidirectional = True
        max_len = 50
        encoder = EncoderRNN(len(input_vocab), max_len, hidden_size,
                             bidirectional=bidirectional, variable_lengths=True)

        decoder = DecoderRNN(len(output_vocab), max_len, hidden_size * 2 if bidirectional else 1,
                             dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                             eos_id=train.tgt_field.eos_id, sos_id=train.tgt_field.sos_id)
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq = seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        scheduler = StepLR(optimizer.optimizer, 1)
        optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=32,
                          checkpoint_every=50,
                          print_every=10, expt_dir=opt.expt_dir)
    start = time.clock()
    seq2seq = t.train(seq2seq, train,
                      num_epochs=6, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)
    end = time.clock() - start
    print('Training time: {:.2f}s'.format(end))

predictor = Predictor(seq2seq, input_vocab, output_vocab)

while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
