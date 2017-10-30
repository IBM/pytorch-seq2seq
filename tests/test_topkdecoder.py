import unittest

import torch
import torch.nn.functional as F
import numpy as np

from seq2seq.models import DecoderRNN, TopKDecoder

class TestDecoderRNN(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.vocab_size = 3

    def test_init(self):
        decoder = DecoderRNN(self.vocab_size, 50, 16, 0, 1, input_dropout_p=0)
        TopKDecoder(decoder, 3)

    def test_k_1(self):
        """ When k=1, the output of topk decoder should be the same as a normal decoder. """
        batch_size = 1
        eos = 1

        for _ in range(10):
            # Repeat the randomized test multiple times
            decoder = DecoderRNN(self.vocab_size, 50, 16, 0, eos)
            for param in decoder.parameters():
                param.data.uniform_(-1, 1)
            topk_decoder = TopKDecoder(decoder, 1)

            output, _, other = decoder()
            output_topk, _, other_topk = topk_decoder()

            self.assertEqual(len(output), len(output_topk))

            finished = [False] * batch_size
            seq_scores = [0] * batch_size

            for t_step, t_output in enumerate(output):
                score, _ = t_output.topk(1)
                symbols = other['sequence'][t_step]
                for b in range(batch_size):
                    seq_scores[b] += score[b].data[0]
                    symbol = symbols[b].data[0]
                    if not finished[b] and symbol == eos:
                        finished[b] = True
                        self.assertEqual(other_topk['length'][b], t_step + 1)
                        self.assertTrue(np.isclose(seq_scores[b], other_topk['score'][b][0]))
                    if not finished[b]:
                        symbol_topk = other_topk['topk_sequence'][t_step][b].data[0][0]
                        self.assertEqual(symbol, symbol_topk)
                        self.assertTrue(torch.equal(t_output.data, output_topk[t_step].data))
                if sum(finished) == batch_size:
                    break

    def test_k_greater_then_1(self):
        """ Implement beam search manually and compare results from topk decoder. """
        max_len = 50
        beam_size = 3
        batch_size = 1
        hidden_size = 8
        sos = 0
        eos = 1

        for _ in range(10):
            decoder = DecoderRNN(self.vocab_size, max_len, hidden_size, sos, eos)
            for param in decoder.parameters():
                param.data.uniform_(-1, 1)
            topk_decoder = TopKDecoder(decoder, beam_size)

            encoder_hidden = torch.autograd.Variable(torch.randn(1, batch_size, hidden_size))
            _, _, other_topk = topk_decoder(encoder_hidden=encoder_hidden)

            # Queue state:
            #   1. time step
            #   2. symbol
            #   3. hidden state
            #   4. accumulated log likelihood
            #   5. beam number
            batch_queue = [[(-1, sos, encoder_hidden[:,b,:].unsqueeze(1), 0, None)] for b in range(batch_size)]
            time_batch_queue = [batch_queue]
            batch_finished_seqs = [list() for _ in range(batch_size)]
            for t in range(max_len):
                new_batch_queue = []
                for b in range(batch_size):
                    new_queue = []
                    for k in range(min(len(time_batch_queue[t][b]), beam_size)):
                        _, inputs, hidden, seq_score, _ = time_batch_queue[t][b][k]
                        if inputs == eos:
                            batch_finished_seqs[b].append(time_batch_queue[t][b][k])
                            continue
                        inputs = torch.autograd.Variable(torch.LongTensor([[inputs]]))
                        decoder_outputs, hidden, _ = decoder.forward_step(inputs, hidden, None, F.log_softmax)
                        topk_score, topk = decoder_outputs[0].data.topk(beam_size)
                        for score, sym in zip(topk_score.tolist()[0], topk.tolist()[0]):
                            new_queue.append((t, sym, hidden, score + seq_score, k))
                    new_queue = sorted(new_queue, key=lambda x: x[3], reverse=True)[:beam_size]
                    new_batch_queue.append(new_queue)
                time_batch_queue.append(new_batch_queue)

            # finished beams
            finalist = [l[:beam_size] for l in batch_finished_seqs]
            # unfinished beams
            for b in range(batch_size):
                if len(finalist[b]) < beam_size:
                    last_step = sorted(time_batch_queue[-1][b], key=lambda x: x[3], reverse=True)
                    finalist[b] += last_step[:beam_size - len(finalist[b])]

            # back track
            topk = []
            for b in range(batch_size):
                batch_topk = []
                for k in range(beam_size):
                    seq = [finalist[b][k]]
                    prev_k = seq[-1][4]
                    prev_t = seq[-1][0]
                    while prev_k is not None:
                        seq.append(time_batch_queue[prev_t][b][prev_k])
                        prev_k = seq[-1][4]
                        prev_t = seq[-1][0]
                    batch_topk.append([s for s in reversed(seq)])
                topk.append(batch_topk)

            for b in range(batch_size):
                topk[b] = sorted(topk[b], key=lambda s: s[-1][3], reverse=True)

            topk_scores = other_topk['score']
            topk_lengths = other_topk['topk_length']
            topk_pred_symbols = other_topk['topk_sequence']
            for b in range(batch_size):
                precision_error = False
                for k in range(beam_size - 1):
                    if np.isclose(topk_scores[b][k], topk_scores[b][k+1]):
                        precision_error = True
                        break
                if precision_error:
                    break
                for k in range(beam_size):
                    self.assertEqual(topk_lengths[b][k], len(topk[b][k]) - 1)
                    self.assertTrue(np.isclose(topk_scores[b][k], topk[b][k][-1][3]))
                    total_steps = topk_lengths[b][k]
                    for t in range(total_steps):
                        self.assertEqual(topk_pred_symbols[t][b, k].data[0], topk[b][k][t+1][1]) # topk includes SOS
