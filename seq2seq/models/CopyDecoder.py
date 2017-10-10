# Reference: https://arxiv.org/pdf/1704.04368.pdf
# For the copy decoder, the target vocabulary is different for each input sequence,
# and thus the input batch has to carry the following information:
#   1. A unique vocab for each input sequence
#   2. Index to retrieve the vocab for each input sequence
#   3. Use the vocab for loss compute and decoding

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DecoderRNN import Decoder

# TODO: abstract decoders with a parent module
class CopyDecoder(Decoder):

    def __init__(self, hidden_size, output_size):
        super(CopyDecoder, self).__init__(hidden_size, output_size)
        self.gen_linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden, attn):
        gen_prob = F.sigmoid(self.gen_linear(hidden.view(-1, self.hidden_size))).log()

        vocab_prob = super(CopyDecoder, self).forward(hidden, attn)
        vocab_prob += gen_prob

        copy_prob = attn.view(-1, attn.size(2)).log() + (1 - gen_prob)

        out_prob = torch.cat([vocab_prob, copy_prob], dim=1)

        return out_prob
