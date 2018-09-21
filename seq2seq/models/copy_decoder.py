from __future__ import print_function

import torch
import torch.nn as nn

from .simple_decoder import SimpleDecoder

class CopyDecoder(SimpleDecoder):
    """ Reference: https://arxiv.org/abs/1704.04368
    Get To The Point: Summarization with Pointer-Generator Networks (See et al.)"""

    def __init__(self, hidden_size, output_size):
        super(CopyDecoder, self).__init__(hidden_size, output_size)
        self.linear_gen = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, context, attn, batch, dataset):
        batch_size = batch.batch_size
        decode_len = context.size(1)
        # pointer
        p_gen = self.sigmoid(self.linear_gen(context.view(-1, self.hidden_size))).log()
        # getting the context and symbols from SimpleDecoder
        prob, symbols = super(CopyDecoder, self).forward(context, attn)
        # prob of sampling a word from target vocab
        p_vocab = prob.view(-1, self.output_size) * p_gen
        # prob of copying a word from source doc
        p_copy = attn.view(-1, attn.size(2)) * (1 - p_gen)
        # total prob on which to predict the sequence
        p_out = torch.cat([p_vocab, p_copy], dim=1)
        # dynamic vocab containing target doc and 
        # mapping from words in source doc to source idx  
        dynamic_vocab = dataset.dynamic_vocab
        tgt_vocab = dataset.fields['tgt'].vocab
        offset = len(tgt_vocab)
        for b in range(batch_size):
            src_idx = batch.index[b].item()
            src_vocab = dynamic_vocab[src_idx]
            src_indices = batch.src_index[b].data.tolist()
            for src_tok_id in src_indices:
                tok = src_vocab.itos[src_tok_id]
                tgt_tok_id = tgt_vocab.stoi[tok]
                if tgt_tok_id != 0:
                    copy_id = offset + src_tok_id - 1
                    p_out[b: b + decode_len, tgt_tok_id] = p_out[b: b + decode_len, tgt_tok_id] \
                        + p_out[b: b + decode_len, copy_id]
                    p_out[b: b + decode_len, copy_id].data.fill_(1e-20)

        # generator
        symbols = p_out.topk(1, dim=1)[1]

        # TODO(1): fix dimensions
        # symbols = symbols.view(batch.batch_size, decode_len)
        # p_out = p_out.view(batch.batch_size, decode_len, -1)
        return p_out, symbols