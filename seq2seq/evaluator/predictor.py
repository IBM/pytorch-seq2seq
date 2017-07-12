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
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        src_id_seq = self.src_vocab.indices_from_sequence(src_seq)

        softmax_list, _, other = self.model([src_id_seq], volatile=True)
        length = other['length'][0]

        tgt_id_seq = []
        for i in range(length):
            idx = softmax_list[i].max(1)[1].data[0][0]
            tgt_id_seq.append(idx)

        tgt_seq = self.tgt_vocab.sequence_from_indices(tgt_id_seq)
        return tgt_seq
