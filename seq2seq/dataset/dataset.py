import codecs

import torch
import torchtext

from . import SourceField, TargetField
from .. import src_field_name, tgt_field_name

def make_example(line, fields):
    pass

def _read_corpus(path):
    with codecs.open(path, 'r', 'utf-8') as fin:
        for line in fin:
            yield line

class Seq2SeqDataset(torchtext.data.Dataset):

    def __init__(self, src_path, tgt_path=None,
                 src_field=SourceField(), tgt_field=TargetField(),
                 **kwargs):
        self.src_field = src_field
        self.tgt_field = tgt_field

        # construct fields
        fields = [(src_field_name, src_field)]
        if tgt_path is not None:
            fields.append((tgt_field_name, tgt_field))
        idx_field = torchtext.data.Field(use_vocab=False,
                                         tensor_type=torch.LongTensor,
                                         sequential=False)
        fields.append(('index', idx_field))

        # construct examples
        corpus = _read_corpus(src_path)
        if tgt_path is not None:
            corpus = zip(corpus, _read_corpus(tgt_path))

        examples = [torchtext.data.Example.fromlist(list(data) + [i], fields)
                    for i, data in enumerate(corpus)]

        super(Seq2SeqDataset, self).__init__(examples, fields, **kwargs)

    def build_vocab(self, src_vocab_size, tgt_vocab_size):
        self.src_field.build_vocab(self, max_size=src_vocab_size)
        self.tgt_field.build_vocab(self, max_size=tgt_vocab_size)
