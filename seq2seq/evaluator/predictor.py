import torch
from torch.autograd import Variable
import torchtext

from seq2seq.data import Seq2SeqDataset

class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of input tokens in source language

        Returns:
            tgt_seq (list): list of output tokens in target language as predicted
            by the pre-trained model
        """
        with torch.no_grad():
            src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] 
                                    for tok in src_seq])).view(1, -1)
            if torch.cuda.is_available():
                src_id_seq = src_id_seq.cuda()

            dataset = Seq2SeqDataset.from_list(' '.join(src_seq))
            dataset.vocab = self.src_vocab
            batch = torchtext.data.Batch.fromvars(dataset, 1, 
                        src=(src_id_seq, [len(src_seq)]), tgt=None)

            _, _, other = self.model(batch)
            
        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq