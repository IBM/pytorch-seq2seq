import codecs
from collections import Counter

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
    """ The idea of dynamic vocabulary is bought from [Opennmt-py](https://github.com/OpenNMT/OpenNMT-py)"""

    def __init__(self, examples, src_field, tgt_field=None, dynamic=True, **kwargs):

        # construct fields
        self.src_field = src_field
        self.tgt_field = tgt_field
        self.fields = [(src_field_name, src_field)]
        if tgt_field is not None:
            self.fields.append((tgt_field_name, tgt_field))

        self.dynamic = dynamic
        self.dynamic_vocab = []
        if self.dynamic:
            src_index_field = torchtext.data.Field(use_vocab=False,
                                                   pad_token=0, sequential=True,
                                                   batch_first=True)
            self.fields.append(('src_index', src_index_field))
            examples = self._add_dynamic_vocab(examples)

        idx_field = torchtext.data.Field(use_vocab=False,
                                         sequential=False)
        self.fields.append(('index', idx_field))
        # construct examples
        examples = [torchtext.data.Example.fromlist(list(data) + [i], self.fields)
                    for i, data in enumerate(examples)]


        super(Seq2SeqDataset, self).__init__(examples, self.fields, **kwargs)

    def _add_dynamic_vocab(self, examples):
        tokenize = self.fields[0][1].tokenize # Tokenize function of the source field
        for example in examples:
            src_seq = tokenize(example[0])
            dy_vocab = torchtext.vocab.Vocab(Counter(src_seq), specials=[])
            self.dynamic_vocab.append(dy_vocab)
            # src_indices = torch.LongTensor([dy_vocab.stoi[w] for w in tokenize(src_seq)])
            src_indices = [dy_vocab.stoi[w] for w in src_seq]
            yield tuple(list(example) + [src_indices])

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
