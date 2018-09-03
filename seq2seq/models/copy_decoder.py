from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .simple_decoder import SimpleDecoder

class CopyDecoder(SimpleDecoder):
    """ Reference: https://arxiv.org/abs/1704.04368
    Get To The Point: Summarization with Pointer-Generator Networks (See et al.)"""

    def __init__(self, hidden_size, output_size):
        super(CopyDecoder, self).__init__(hidden_size, output_size)
        self.linear_gen = nn.Linear(hidden_size, 1)
        self.coverage_vec = None

    def forward(self, context, attn, batch, dataset):
        decode_len = context.size(1)
        self.coverage_vec = self._coverage(attn)

        p_gen = F.sigmoid(self.linear_gen(context.view(-1, self.hidden_size))).log()

        p_vocab, symbols = super(CopyDecoder, self).forward(context, attn)
        p_vocab = p_vocab.view(-1, self.output_size) * p_gen
        p_copy = attn.view(-1, attn.size(2)) * (1 - p_gen)
        p_out = torch.cat([p_vocab, p_copy], dim=1)

        dynamic_vocab = dataset.dynamic_vocab
        tgt_vocab = dataset.fields['tgt'].vocab
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
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

        symbols = p_out.topk(1, dim=1)[1]
        return p_out, symbols, self.coverage_vec

    def _copy(self):
        """ Implement copy mechanism """
        pass

    def _coverage(self, attn):
        """ Implement coverage mechanism """
        return self.coverage_vec + attn