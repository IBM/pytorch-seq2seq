from __future__ import print_function

import torch
import torch.nn as nn

from .DecoderRNN import Decoder

# TODO: abstract decoders with a parent module
class CopyDecoder(Decoder):
    """ Reference: https://arxiv.org/pdf/1704.04368.pdf """


    def __init__(self, hidden_size, output_size):
        super(CopyDecoder, self).__init__(hidden_size, output_size)
        self.gen_linear = nn.Linear(hidden_size, 1)

    def forward(self, context, attn, batch, dataset):
        de_len = context.size(1)
        gen_prob = torch.nn.sigmoid(self.gen_linear(context.view(-1, self.hidden_size))).log()

        vocab_prob, symbols = super(CopyDecoder, self).forward(context, attn)
        vocab_prob = vocab_prob.view(-1, self.output_size) * gen_prob

        copy_prob = attn.view(-1, attn.size(2)) * (1 - gen_prob)

        out_prob = torch.cat([vocab_prob, copy_prob], dim=1)

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
                    out_prob[b: b + de_len, tgt_tok_id] = out_prob[b: b + de_len, tgt_tok_id] \
                        + out_prob[b: b + de_len, copy_id]
                    out_prob[b: b + de_len, copy_id].data.fill_(1e-20)

        symbols = out_prob.topk(1, dim=1)[1]

        return out_prob, symbols
