import unittest
from seq2seq.dataset.vocabulary import Vocabulary
import cPickle as pickle

class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary(50000)

    ######################################################################
    #  get_index(token)
    ######################################################################
    def test_get_index_WITH_VALID_TOKEN(self):
        self.assertEqual(0, self.vocab.get_index("MASK"))

    def test_get_index_WITH_INVALID_TOKEN(self):
        self.assertRaises(LookupError, self.vocab.get_index, "python")

    ######################################################################
    #  get_token(index)
    ######################################################################
    def test_get_token_WITH_VALID_INDEX(self):
        self.assertEqual("EOS", self.vocab.get_token(2))

    def test_get_token_WITH_INVALID_INDEX(self):
        self.assertRaises(LookupError, self.vocab.get_index, 5)

    ######################################################################
    #  get_vocab_size(index)
    ######################################################################
    def test_get_vocab_size(self):
        self.assertEqual(3, self.vocab.get_vocab_size())

    ######################################################################
    #  add_token(token)
    ######################################################################
    def test_add_token_WITH_NEW_TOKEN(self):
        self.assertEqual(3, self.vocab.get_vocab_size())
        self.vocab.add_token("python")
        self.assertEqual(4, self.vocab.get_vocab_size())
        self.assertEqual(3, self.vocab.get_index("python"))

    def test_add_token_WITH_EXISTING_TOKEN(self):
        self.vocab.add_token("abc")
        self.assertEqual(4, self.vocab.get_vocab_size())
        self.vocab.add_token("abc")
        self.assertEqual(4, self.vocab.get_vocab_size())

    ######################################################################
    #  add_sequence(sequence)
    ######################################################################
    def test_add_sequence_WITH_NEW_SEQUENCE(self):
        self.assertEqual(3, self.vocab.get_vocab_size())
        self.vocab.add_sequence(["i", "like", "python"])
        self.assertEqual(6, self.vocab.get_vocab_size())

    def test_add_sequence_WITH_PARTIAL_NEW_SEQUENCE(self):
            self.assertEqual(3, self.vocab.get_vocab_size())
            self.vocab.add_sequence(["i", "like", "python", "EOS"])
            self.vocab.add_sequence(["i"])
            self.assertEqual(6, self.vocab.get_vocab_size())
            self.assertEqual(3, self.vocab.get_index('i'))

    ######################################################################
    #  indices_from_sequence(sequence)
    ######################################################################
    def test_indices_from_sequence_WITH_NEW_SEQUENCE(self):
        self.vocab.add_sequence(["i", "like", "python"])
        self.assertSetEqual(set([3, 4, 5]), set(self.vocab.indices_from_sequence(["i", "like", "python"])))

    def test_indices_from_sequence_WITH_PARTIAL_NEW_SEQUENCE(self):
        self.vocab.add_sequence(["i", "like", "python", "EOS"])
        self.assertSetEqual(set([3, 4, 5, 2]), set(self.vocab.indices_from_sequence(["i", "like", "python", "EOS"])))

    def test_indices_from_sequence_WITH_OUT_OF_VOCAB_TOKEN(self):
        self.vocab.add_sequence(["i", "like", "python", "EOS"])
        self.assertSetEqual(set([3, 5, 0, 2]), set(self.vocab.indices_from_sequence(["i", "like", "java", "EOS"])))

    def test_indices_from_sequence_WITH_SMALL_VOCAB_SIZE(self):
        vocab = Vocabulary(3)
        vocab.add_sequence(["i", "like", "python", "EOS"])
        vocab.add_sequence(["i", "also", "like", "java"])
        vocab.add_sequence(["some", "people", "like", "C++", "EOS"])
        self.assertSetEqual(set([3, 4, 0, 2]), set(vocab.indices_from_sequence(["i", "like", "python", "EOS"])))

    ######################################################################
    #  sequence_from_indices(indices)
    ######################################################################
    def test_sequence_from_indices(self):
        vocab = self.vocab
        seq = ["i", "like", "python"]
        vocab.add_sequence(seq)
        indices = [vocab.get_index(tok) for tok in seq]
        sequence = vocab.sequence_from_indices(indices)
        self.assertEqual(seq, sequence)


    ######################################################################
    #  save(file_name)
    ######################################################################
    def test_save_vocab(self):
        vocab = self.vocab
        seq = ["i", "like", "python"]
        vocab.add_sequence(seq)
        vocab.save("vocab_pickle")
        with open("vocab_pickle","rb") as f:
            loaded_vocab = pickle.load(f)
        self.assertEqual(vocab, loaded_vocab)

    ######################################################################
    #  load(file_name)
    ######################################################################
    def test_load(self):
        vocab = self.vocab
        seq = ["i", "like", "python"]
        vocab.add_sequence(seq)
        pickle_file = "vocab_pickle"
        input_vocab_pickle = pickle.dumps(vocab)
        with open(pickle_file, "wb") as f:
            f.write(input_vocab_pickle)

        with open(pickle_file,"rb") as f:
            pickled_vocab = pickle.load(f)

        loaded_vocab = Vocabulary.load(pickle_file)
        self.assertEqual(pickled_vocab, loaded_vocab)

    ######################################################################
    #  __eq__(self, other)
    ######################################################################
    def testing_one_vocab_subset_of_another(self):
        vocab = self.vocab
        seq = ["i", "like", "python"]
        vocab.add_sequence(seq)

        other_vocab = Vocabulary(50000)
        other_seq = ["i", "like", "python", "too"]
        other_vocab.add_sequence(other_seq)

        self.assertNotEqual(vocab, other_vocab)

    ######################################################################
    #  __eq__(self, other)
    ######################################################################
    def testing_vocab_of_same_sequence(self):
        vocab = self.vocab
        seq = ["i", "like", "python"]
        vocab.add_sequence(seq)

        other_vocab = Vocabulary(50000)
        other_seq = ["i", "like", "python"]
        other_vocab.add_sequence(other_seq)

        self.assertEqual(vocab, other_vocab)

    ######################################################################
    #  __eq__(self, other)
    ######################################################################
    def testing_vocab_of_different_order(self):
        vocab = self.vocab
        seq = ["i", "like", "like", "python"]
        vocab.add_sequence(seq)

        other_vocab = Vocabulary(50000)
        other_seq = ["i", "like", "python", "like"]
        other_vocab.add_sequence(other_seq)

        self.assertEqual(vocab, other_vocab)

######################################################################
#   M A I N
######################################################################
if __name__ == '__main__':
    unittest.main()
