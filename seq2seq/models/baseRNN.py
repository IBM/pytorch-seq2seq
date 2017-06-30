""" A base class for RNN. """
import torch
import torch.nn as nn


class BaseRNN(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence.
    Note:
        Do not use this class directly, use one of the sub classes.
    Args:
        vocab (Vocabulary): object of Vocabulary class
        max_len (int): maximum allowed length for the sequence to be processed
        hidden_size (int): number of features in the hidden state `h`
        input_dropout_p (float): dropout probability for the input sequence
        dropout_p (float): dropout probability for the output sequence
        n_layers (int): number of recurrent layers
        rnn_cell (str): type of RNN cell (Eg. 'LSTM' , 'GRU')

    Inputs: ``*args``, ``**kwargs``
        - ``*args``: variable length argument list.
        - ``**kwargs``: arbitrary keyword arguments.

    Attributes:
        SYM_MASK: masking symbol
        SYM_EOS: end-of-sequence symbol
    """
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, vocab, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

    def balance(self, batch, volatile):
        """
        Add reserved symbols and balance batch input.
        It first appends EOS symbol to each sequence and then appends multiple
        MASK symbols to make the sequences the same length.
        Args:
            batch: list of sequences, each of which is a list of integers
            volatile: boolean flag specifying whether to preserve gradients, when you are sure you will not be even calling .backward().

        Returns:
            torch.autograd.Variable: variable with balanced input data.
        """
        max_len = self.max_len
        outputs = []
        for seq in batch:
            seq = seq[:min(len(seq), max_len - 1)]
            outputs.append(seq + [self.vocab.EOS_token_id] + [self.vocab.MASK_token_id] * (max_len - len(seq) - 1))

        outputs_var = torch.autograd.Variable(torch.LongTensor(outputs), volatile=volatile)
        if torch.cuda.is_available():
            outputs_var = outputs_var.cuda()

        return outputs_var

    def forward(self, *args, **kwargs):
        if 'volatile' in kwargs:
            volatile = kwargs['volatile']
            kwargs.pop('volatile', None)
        else:
            volatile = False
        if args:
            self.balanced_batch = self.balance(args[0], volatile)
            args = [self.balanced_batch] + list(args[1:])
        else:
            if 'inputs' in kwargs and kwargs['inputs'] is not None:
                self.balanced_batch = self.balance(kwargs['inputs'], volatile)
                kwargs['inputs'] = self.balanced_batch

        return self.forward_rnn(*args, **kwargs)

    def forward_rnn(self, *args, **kwargs):
        raise NotImplementedError()
