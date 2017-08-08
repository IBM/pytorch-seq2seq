import unittest

import torchtext

from seq2seq.dataset import SourceField, TargetField

class TestField(unittest.TestCase):

    def test_sourcefield(self):
        field = SourceField()
        self.assertTrue(isinstance(field, torchtext.data.Field))
        self.assertTrue(field.batch_first)
        self.assertTrue(field.include_lengths)

    def test_sourcefield_with_wrong_setting(self):
        field = SourceField(batch_first=False, include_lengths=False)
        self.assertTrue(isinstance(field, torchtext.data.Field))
        self.assertTrue(field.batch_first)
        self.assertTrue(field.include_lengths)

    def test_targetfield(self):
        field = TargetField()
        self.assertTrue(isinstance(field, torchtext.data.Field))
        self.assertTrue(field.batch_first)

        processed = field.preprocessing([None])
        self.assertEqual(processed, ['<sos>', None, '<eos>'])

    def test_targetfield_with_other_setting(self):
        field = TargetField(batch_first=False, preprocessing=lambda seq: seq + seq)
        self.assertTrue(isinstance(field, torchtext.data.Field))
        self.assertTrue(field.batch_first)

        processed = field.preprocessing([None])
        self.assertEqual(processed, ['<sos>', None, None, '<eos>'])
