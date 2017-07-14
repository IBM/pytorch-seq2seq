import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.
    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, target_variable, teacher_forcing_ratio, volatile
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)
        - **volatile** (bool, optional): boolean flag specifying whether to preserve gradients, when you are sure you
          will not be even calling .backward().

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

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def forward(self, input_variable, target_variable=None, teacher_forcing_ratio=0,
                volatile=False):
        if target_variable is None:
            input_variable = sorted(input_variable, len, reverse=True)
        else:
            sorted_input = sorted(
                zip(input_variable, target_variable), key=lambda x: len(x[0]), reverse=True)
            input_variable = [p[0] for p in sorted_input]
            target_variable = [p[1] for p in sorted_input]
        encoder_outputs, encoder_hidden = self.encoder(input_variable, volatile=volatile)
        result = self.decoder(
            inputs=target_variable,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            function=self.decode_function,
            teacher_forcing_ratio=teacher_forcing_ratio,
            volatile=volatile)
        return result

