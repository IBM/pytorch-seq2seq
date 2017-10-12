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

    def __init__(self, corpus, src_field, tgt_field=None, **kwargs):

        # construct fields
        self.src_field = src_field
        self.tgt_field = tgt_field
        fields = [(src_field_name, src_field)]
        if tgt_field is not None:
            fields.append((tgt_field_name, tgt_field))
        idx_field = torchtext.data.Field(use_vocab=False,
                                         tensor_type=torch.LongTensor,
                                         sequential=False)
        fields.append(('index', idx_field))

        # construct examples
        examples = [torchtext.data.Example.fromlist(list(data) + [i], fields)
                    for i, data in enumerate(corpus)]


        super(Seq2SeqDataset, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def from_file(src_path, tgt_path=None, share_fields_from=None, **kwargs):
        src_list = _read_corpus(src_path)
        if tgt_path is not None:
            tgt_list = _read_corpus(tgt_path)
        else:
            tgt_list = None
        return Seq2SeqDataset.from_list(src_list, tgt_list, share_fields_from, **kwargs)

    @staticmethod
    def from_list(src_list, tgt_list=None, share_fields_from=None, **kwargs):
        corpus = src_list
        if share_fields_from is not None:
            src_field = share_fields_from.fields[src_field_name]
        else:
            src_field = SourceField()
        tgt_field = None
        if tgt_list is not None:
            corpus = zip(corpus, tgt_list)
            if share_fields_from is not None:
                tgt_field = share_fields_from.fields[tgt_field_name]
            else:
                tgt_field = TargetField()
        return Seq2SeqDataset(corpus, src_field, tgt_field, **kwargs)

    def build_vocab(self, src_vocab_size, tgt_vocab_size):
        self.src_field.build_vocab(self, max_size=src_vocab_size)
        self.tgt_field.build_vocab(self, max_size=tgt_vocab_size)
