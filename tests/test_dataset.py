import unittest

from seq2seq.dataset.dataset import Dataset


######################################################################
#  T E S T   C A S E S
######################################################################
class TestDataset(unittest.TestCase):

    def setUp(self):
        self.ds = Dataset("./tests/data/eng-fra.txt", 10, 10,
                src_max_vocab=50000, tgt_max_vocab=50000)

    ######################################################################
    #  __init__()
    ######################################################################
    def test_init(self):
        self.assertEqual(100, self.ds.input_vocab.get_vocab_size())
        self.assertEqual(146, self.ds.output_vocab.get_vocab_size())
        self.assertFalse(set(self.ds.input_vocab._token2index.keys()) == set(self.ds.output_vocab._token2index.keys()))

    def test_init_WITH_VOCABULARY(self):
        new_ds = Dataset("./tests/data/eng-fra-dev.txt", 10, 10,
                         src_vocab=self.ds.input_vocab,
                         tgt_vocab=self.ds.output_vocab)
        self.assertEqual(self.ds.input_vocab, new_ds.input_vocab)
        self.assertEqual(self.ds.output_vocab, new_ds.output_vocab)

        ds_with_new_vocab = Dataset("./tests/data/eng-fra-dev.txt", 10, 10)
        # Vocabulary of the training data should have no overlap with
        # that of the develop data
        input_vocab = set(self.ds.input_vocab._token2index.keys())
        new_input_vocab = set(ds_with_new_vocab.input_vocab._token2count.keys())
        self.assertFalse(input_vocab & new_input_vocab)
        output_vocab = set(self.ds.output_vocab._token2index.keys())
        new_output_vocab = set(ds_with_new_vocab.output_vocab._token2count.keys())
        self.assertFalse(output_vocab & new_output_vocab)


        # Since there's no overlap between vocabularies
        # The dev set loaded with the training vocabulary should be
        # all masked sequences
        for batch in new_ds.make_batches(1):
            src, tgt = batch
            self.assertEquals([0], src[0])
            self.assertEquals([0], tgt[0])

    def test_init_WITH_VOCABULARY_FILE(self):
        new_ds = Dataset("./tests/data/eng-fra-dev.txt", 10, 10,
                         src_vocab="./tests/data/src_vocab.txt",
                         tgt_vocab="./tests/data/tgt_vocab.txt")
        self.assertEqual(8, new_ds.input_vocab.get_vocab_size())
        self.assertEqual(8, new_ds.output_vocab.get_vocab_size())
        self.assertEqual({'good day', 'welcome', 'MASK', 'EOS', 'thank you', 'SOS', 'hi', 'hello'},
                         set(new_ds.input_vocab._token2index.keys()))
        self.assertEqual({'bienvenue', 'MASK', 'EOS', 'SOS', 'bonne journee', 'salut', 'Je vous remercie', 'bonjour'},
                         set(new_ds.output_vocab._token2index.keys()))

        self.assertFalse(set(self.ds.input_vocab._token2index.keys()) == set(new_ds.input_vocab._token2index.keys()))
        self.assertFalse(set(self.ds.output_vocab._token2index.keys()) == set(new_ds.output_vocab._token2index.keys()))

    ######################################################################
    #  make_batches(batch_size)
    ######################################################################
    def test_make_batches_WITH_LARGER_BATCH_SIZE(self):
        with self.assertRaises(OverflowError):
            for batch in self.ds.make_batches(180):
                pass

    def test_make_batches_WITH_EXACT_BATCH_SIZE(self):
        batch_size = 76
        num_data = len(self.ds)
        batches = self.ds.make_batches(batch_size)

        first_batch = next(batches)
        self.assertEqual(2, len(first_batch))
        self.assertEqual(batch_size, len(first_batch[0]))
        self.assertEqual(batch_size, len(first_batch[1]))

        second_batch = next(batches)
        self.assertEqual(2, len(second_batch))
        self.assertEqual(num_data - batch_size, len(second_batch[0]))
        self.assertEqual(num_data - batch_size, len(second_batch[1]))

        self.assertRaises(StopIteration, lambda: next(batches))

    def test_make_batches(self):
        num_batches = 0
        batch_size = 10
        for batch in self.ds.make_batches(10):
            num_batches += 1
            self.assertEqual(2, len(batch))
            self.assertEqual(batch_size, len(batch[0]))
            self.assertEqual(batch_size, len(batch[1]))
        self.assertEqual(batch_size, num_batches)

    ######################################################################
    #  __len__()
    ######################################################################
    def test_len(self):
        self.assertEqual(100, self.ds.__len__())

    ######################################################################
    #  shuffle()
    ######################################################################
    def test_shuffle(self):
        data = self.ds.data[:]
        self.ds.shuffle()
        shuffled_data = self.ds.data
        self.assertFalse(data == shuffled_data)

    @unittest.skip('Need investigation on random.seed() method')
    def test_shuffle_WITH_SAME_SEED(self):
        self.ds.shuffle(seed=123)
        seeded_data = self.ds.data[:]
        self.ds.shuffle()
        shuffled_data = self.ds.data[:]
        self.assertFalse(seeded_data == shuffled_data)
        self.ds.shuffle(seed=123)
        reseeded_data = self.ds.data[:]
        self.assertListEqual(seeded_data, reseeded_data)

######################################################################
#   M A I N
######################################################################
if __name__ == '__main__':
    unittest.main()
