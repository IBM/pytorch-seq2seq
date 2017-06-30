# This Python file uses the following encoding: utf-8
import unittest
from seq2seq.dataset import utils


######################################################################
#  T E S T   C A S E S
######################################################################
class TestDatasetUtils(unittest.TestCase):

    ######################################################################
    #  space_tokenizer(s)
    ######################################################################
    def test_space_tokenize(self):
        test_string = "I like python"
        tokenized_string = utils.space_tokenize(test_string)
        self.assertEquals(3, len(tokenized_string))

    ######################################################################
    #  filterPair(pairs, src_max_len, tgt_max_len)
    ######################################################################
    def test_filterPair_WITH_VALID_PAIR(self):
        test_pair = [["you", "are", "wrong", "."],	["tu", "as", "tort", "."]]
        self.assertTrue(utils.filter_pair(test_pair, 10, 10))

    def test_filterPair_WITH_INVALID_PAIR_LONG_INPUT(self):
        test_pair = [["you", "are", "wrong", "."],	["tu", "as", "tort", "."]]
        self.assertFalse(utils.filter_pair(test_pair, 1, 10))

    def test_filterPair_WITH_INVALID_PAIR_LONG_OUTPUT(self):
        test_pair = [["you", "are", "wrong", "."],	["tu", "as", "tort", "."]]
        self.assertFalse(utils.filter_pair(test_pair, 10, 1))

    ########################################################################################################
    #   read_vocabulary(path, max_vocab)
    ########################################################################################################
    def test_read_vocabulary_WITH_VALID_PATH(self):
        vocab = utils.read_vocabulary("./tests/data/src_vocab.txt")
        self.assertEqual(5, len(vocab))

    def test_read_vocabulary_WITH_VALID_PATH_SMALL_MAX_VOCAB(self):
        vocab = utils.read_vocabulary("./tests/data/src_vocab.txt", 3)
        self.assertEqual(3, len(vocab))

    def test_read_vocabulary_WITH_INVALID_PATH(self):
        self.assertRaises(IOError, utils.read_vocabulary, "blah.txt")

    ########################################################################################################
    #   prepare_data(path, src_max_len, tgt_max_len, tokenize_func=space_tokenize, reverse=False)
    ########################################################################################################
    def test_prepare_data_WITH_VALID_PATH(self):
        pairs = utils.prepare_data("./tests/data/eng-fra.txt", 20, 20)
        self.assertEqual(100, len(pairs))

    def test_prepare_data_WITH_INVALID_PATH(self):
        self.assertRaises(IOError, utils.prepare_data, "eng-fra.txt", 20, 20)

######################################################################
#   M A I N
######################################################################
if __name__ == '__main__':
    unittest.main()
