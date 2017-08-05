import os
import argparse
import logging

import torch
import torchtext

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
parser.add_argument('--log-level', dest='log_level',
                    default='debug',
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
    src = torchtext.data.Field(preprocessing=lambda seq: seq + ['<eos>'], batch_first=True)
    trg = torchtext.data.Field(preprocessing=lambda seq: ['<sos>'] + seq + ['<eos>'], batch_first=True)
    max_len = 50
    def len_filter(example):
        return len(example.src) <= max_len and len(example.trg) <= max_len
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('src', src), ('trg', trg)],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('src', src), ('trg', trg)],
        filter_pred=len_filter
    )
    src.build_vocab(train, max_size=50000)
    trg.build_vocab(train, max_size=50000)

    # Prepare dataset
    # dataset = Dataset.from_file(opt.train_path, src_max_len=50, tgt_max_len=50)
    # input_vocab = dataset.input_vocab
    # output_vocab = dataset.output_vocab

    # dev_set = Dataset.from_file(opt.dev_path, src_max_len=50, tgt_max_len=50,
                    # src_vocab=input_vocab,
                    # tgt_vocab=output_vocab)

    # # Prepare loss
    weight = torch.ones(len(trg.vocab))
    pad = trg.vocab.stoi[trg.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    if not opt.resume:
        # Initialize model
        hidden_size=128
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size)
        decoder = DecoderRNN(len(trg.vocab), max_len, hidden_size,
                             dropout_p=0.2, use_attention=True,
                             eos_id=trg.vocab.stoi['<eos>'])
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=32,
                        checkpoint_every=50,
                        print_every=4, expt_dir=opt.expt_dir)
    t.train(seq2seq, train, num_epochs=1, dev_data=dev, resume=opt.resume)

predictor = Predictor(seq2seq, src.vocab, trg.vocab)

while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.split()
    print(predictor.predict(seq))
