import unittest
import os
import shutil

from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.dataset import Dataset
class TestSeq2seq(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_wd = os.getcwd()
        self.dataset = Dataset(path=os.path.join(self.test_wd,'tests/data/eng-fra.txt'),
                               src_max_len=50, tgt_max_len=50, src_max_vocab=50000, tgt_max_vocab=50000)
        self.encoder = EncoderRNN(self.dataset.input_vocab,max_len=10, hidden_size=10, rnn_cell='lstm')
        self.decoder = DecoderRNN(self.dataset.output_vocab, max_len=10, hidden_size=10, rnn_cell='lstm')
        self.seq2seq = Seq2seq(self.encoder,self.decoder)
        self.mock_seq2seq = Seq2seq(self.encoder, self.decoder)

        for param in self.seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        if not os.path.exists(os.path.join(self.test_wd,'checkpoints')):
            os.mkdir(os.path.join(self.test_wd,'checkpoints'))

        self.seq2seq.save(os.path.join(self.test_wd,'checkpoints'))
        self.mock_seq2seq.load(os.path.join(self.test_wd, 'checkpoints'))



    def test_save(self):
        self.assertTrue(os.path.isfile(os.path.join(self.test_wd,'checkpoints','encoder')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_wd, 'checkpoints', 'decoder')))

    def test_load(self):
        self.assertTrue(self.seq2seq.encoder.state_dict() == self.mock_seq2seq.encoder.state_dict())
        self.assertTrue(self.seq2seq.decoder.state_dict() == self.mock_seq2seq.decoder.state_dict())

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(os.path.join(self.test_wd, 'checkpoints'))

if __name__ == '__main__':
    unittest.main()
