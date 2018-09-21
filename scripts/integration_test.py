import os
import time
import argparse
import logging

import torch

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, TopKDecoder, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.data import Seq2SeqDataset
from seq2seq.evaluator import Predictor, Evaluator
from seq2seq.util.checkpoint import Checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--train_src', action='store', help='Path to train source data')
parser.add_argument('--train_tgt', action='store', help='Path to train target data')
parser.add_argument('--dev_src', action='store', help='Path to dev source data')
parser.add_argument('--dev_tgt', action='store', help='Path to dev target data')
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

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 128
        bidirectional = True
        max_len = 50
        encoder = EncoderRNN(len(input_vocab), max_len, hidden_size,
                             bidirectional=bidirectional, rnn_cell='lstm',
                             variable_lengths=True)

        decoder = DecoderRNN(len(output_vocab), max_len, hidden_size * 2,
                             dropout_p=0.2, use_attention=True,
                             bidirectional=bidirectional,rnn_cell='lstm',
                             eos_id=train.tgt_field.eos_id, sos_id=train.tgt_field.sos_id)
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq = seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=32,
                          checkpoint_every=50,
                          print_every=10, experiment_directory=opt.expt_dir)
    start = time.clock()
    seq2seq = t.train(seq2seq, train,
                      n_epochs=6, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)
    end = time.clock() - start
    print('Training time: {:.2f}s'.format(end))

evaluator = Evaluator(loss=loss, batch_size=32)
dev_loss, accuracy = evaluator.evaluate(seq2seq, dev)
assert dev_loss < 1.5

beam_search = Seq2seq(seq2seq.encoder, TopKDecoder(seq2seq.decoder, 3))

predictor = Predictor(beam_search, input_vocab, output_vocab)
inp_seq = "1 3 5 7 9"
seq = predictor.predict(inp_seq.split())
assert " ".join(seq[:-1]) == inp_seq[::-1]
