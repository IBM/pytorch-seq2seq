import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from attention import Attention
from baseRNN import BaseRNN


class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab (Vocabulary): an object of Vocabulary class
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`
        KEY_INPUT (str): key used to target outputs in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (seq_len, batch, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default is `None`)
        - **encoder_hidden** (batch, seq_len, hidden_size): tensor containing the features in the hidden state `h` of
          encoder. Used as the initial hidden state of the decoder.
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    KEY_INPUT = 'inputs'

    def __init__(self,
                 vocab,
                 max_len,
                 hidden_size,
                 n_layers=1,
                 rnn_cell='gru',
                 input_dropout_p=0,
                 dropout_p=0,
                 use_attention=False):
        super(DecoderRNN, self).__init__(vocab, max_len, hidden_size, input_dropout_p, dropout_p,
                                         n_layers, rnn_cell)

        self.output_size = self.vocab.get_vocab_size()
        self.dropout_p = dropout_p
        self.max_length = max_len
        self.use_attention = use_attention

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def init_start_input(self, batch_size):
        # GO input for decoder # Re-initialize when batch size changes
        if self.init_input is None or self.init_input.size(0) != batch_size:
            self.init_input = Variable(
                torch.LongTensor([[self.vocab.SOS_token_id] * batch_size])).view(batch_size, -1)
            if torch.cuda.is_available():
                self.init_input = self.init_input.cuda()
        return self.init_input

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)
        output = self.dropout(output)
        output = output.squeeze(1)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output))
        return predicted_softmax, hidden, attn

    def forward_rnn(self,
                    inputs=None,
                    encoder_hidden=None,
                    function=F.log_softmax,
                    encoder_outputs=None,
                    teacher_forcing_ratio=0):
        ret_dict = dict()
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError(
                    "Teacher forcing has to be disabled (set 0) when no inputs is provided.")
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        decoder_input = self.init_start_input(batch_size)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = [self.max_length] * batch_size

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        h_t = []
        for di in range(self.max_length):
            decoder_output, decoder_hidden, attn = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, function=function)
            decoder_outputs.append(decoder_output)
            h_t.append(decoder_hidden)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(attn)

            symbols = decoder_output.topk(1)[1]
            sequence_symbols.append(symbols)
            eos_batches = symbols.data.eq(self.vocab.EOS_token_id).nonzero()
            if eos_batches.dim() > 0:
                for b_idx in eos_batches[:, 0]:
                    if di < lengths[b_idx]:
                        lengths[b_idx] = di + 1

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                decoder_input = inputs[:, di].contiguous().view(batch_size, 1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths
        ret_dict[DecoderRNN.KEY_INPUT] = inputs

        return decoder_outputs, decoder_hidden, ret_dict
