import random
from seq2seq.dataset import Vocabulary, utils


class Dataset(object):
    """
    A class that encapsulates a dataset of sequence.
    Initialize a dataset from the file at given path.

    Note: - The file must contains a list of TAB-separated pairs of sequences.

          - Source or target sequences that are longer than the respective
            max length will be filtered.

          - As specified by maximum vocabulary size, source and target
            vocabularies will be sorted in descending token frequency and cutoff.
            @TODO This is a hardcoded pre-processing decision. We should conisder
            abstracting  it out.

          - Tokens that are in the dataset but not retained in the vocabulary
            will be dropped in the sequences.

    Args:
        path (str): path to the dataset file
        src_max_len (int): maximum source sequence length
        tgt_max_len (int): maximum target sequence length
        src_vocab (Vocabulary): pre-populated Vocabulary object or a path of a file containing words for the source language,
            default `None`. If a pre-populated Vocabulary object, `src_max_vocab` wouldn't be used.
        tgt_vocab (Vocabulary): pre-populated Vocabulary object or a path of a file containing words for the target language,
            default `None`. If a pre-populated Vocabulary object, `tgt_max_vocab` wouldn't be used.
        src_max_vocab (int): maximum source vocabulary size
        tgt_max_vocab (int): maximum target vocabulary size
    """

    def __init__(self,
                 path,
                 src_max_len,
                 tgt_max_len,
                 src_vocab=None,
                 tgt_vocab=None,
                 src_max_vocab=50000,
                 tgt_max_vocab=50000):
        # Prepare data
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        pairs = utils.prepare_data(path, src_max_len, tgt_max_len)

        # Read in vocabularies
        self.input_vocab = self._init_vocab(zip(*pairs)[0], src_max_vocab, src_vocab)
        self.output_vocab = self._init_vocab(zip(*pairs)[1], tgt_max_vocab, tgt_vocab)

        # Translate input sequences to token ids
        self.data = []
        for pair in pairs:
            src = self.input_vocab.indices_from_sequence(pair[0])
            dst = self.output_vocab.indices_from_sequence(pair[1])
            self.data.append((src, dst))

    def _init_vocab(self, sequences, max_num_vocab, vocab):
        """
        Initiate a vocabulary based off a list of sequences.

        @TODO: This init is overloaded with multiple behavior for convenience
               but poor modularity of function. Consider splitting.

        Args:
            sequence (Iterable)
                S

            max_num_vocab (int)
                 Maximum number of entries allowed in the vocabulary.
                 Trimmed by frequency.

            vocab (str, Vocabulary)
                IF Vocabulary: overrides the incoming sequence data by
                               simply returning that vocab as is.

                IF str: assumes it's in a vocabulary format and read
                        accordingly.

        Returns:
            resp_vocab (Vocabulary)
                Vocabulary object.
        """
        resp_vocab = Vocabulary(max_num_vocab)

        if vocab is None:
            # Build vocabulary from the sequence token
            for sequence in sequences:
                resp_vocab.add_sequence(sequence)

            # Sorts by frequency and trims to max_num_vocab
            resp_vocab.trim()

        elif isinstance(vocab, Vocabulary):
            resp_vocab = vocab

        elif isinstance(vocab, str):
            # Read vocabulary from predefined forma
            for tok in utils.read_vocabulary(vocab, max_num_vocab):
                resp_vocab.add_token(tok)
        else:
            raise AttributeError(
                '{} is not a valid instance on a vocabulary. None, instance of Vocabulary class \
                                 and str are only supported formats for the vocabulary'
                .format(vocab))

        return resp_vocab

    def __len__(self):
        return len(self.data)

    def num_batches(self, batch_size):
        """
        Get the number of batches given batch size.

        Args:
            batch_size(int): number of examples in a batch

        Returns:
            int: number of batches
        """
        return len(range(0, len(self.data), batch_size))

    def make_batches(self, batch_size):
        """
        Create a generator that generates batches in batch_size over data.

        Args:
            batch_size (int): number of pairs in a mini-batch

        Yields:
            (list(str), list(str)): next pair of source and target variable in a batch

        """
        if len(self.data) < batch_size:
            raise OverflowError("batch size = {} cannot be larger than data size = {}".format(
                batch_size, len(self.data)))

        # Yields data for every basic_size
        for i in range(0, len(self.data), batch_size):
            cur_batch = self.data[i:i + batch_size]
            source_variables = [pair[0] for pair in cur_batch]
            target_variables = [pair[1] for pair in cur_batch]

            yield (source_variables, target_variables)

    def shuffle(self, seed=None):
        """
        Shuffle the data.

        Args:
            seed(int): provide a value for the random seed; default seed=None is truly random
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
