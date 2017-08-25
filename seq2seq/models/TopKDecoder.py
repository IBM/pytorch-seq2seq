import torch
import torch.nn.functional as F
from torch.autograd import Variable
from .baseRNN import BaseRNN


class TopKDecoder(BaseRNN):
    r"""
    Top-K decoding with beam search.

    Args:
        decoder_rnn (DecoderRNN): An object of DecoderRNN used for decoding.
        k (int): Size of the beam.

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (seq_len, batch, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default is `None`)
        - **encoder_hidden** (batch, seq_len, hidden_size): tensor containing the features in the hidden state `h` of
          encoder. Used as the initial hidden state of the decoder.
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*length* : list of integers
          representing lengths of output sequences, *sequence* : list of sequences, where each sequence is a list of
          predicted token IDs, *inputs* : target outputs if provided for decoding}.

    """

    def __init__(self, decoder_rnn, k):
        super(TopKDecoder, self).__init__(decoder_rnn.lang, decoder_rnn.max_length)
        self.rnn = decoder_rnn
        self.k = k
        self.V = self.rnn.lang.get_vocab_size()
        self.SOS = self.rnn.lang.SOS_token_id
        self.EOS = self.rnn.lang.EOS_token_id

    def forward_rnn(self, inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax,
                    retain_output_probs=True):
        """
        Forward rnn for MAX_LENGTH steps.  Look at :func:`seq2seq.models.DecoderRNN.DecoderRNN.forward_rnn` for details.
        """

        # TODO: Looks like encoder_hidden is not optional, we need unit tests
        # for this class
        # Get batch size, assuming h_0 is num_layers*directions x b x hidden_dim
        b = encoder_hidden.size(1)
        h = encoder_hidden.size(2)

        self.pos_index = Variable(torch.LongTensor(range(b)) * self.k).view(-1, 1)

        # Inflate the initial hidden states to be of size: b*k x h
        hidden = self._inflate(encoder_hidden, self.k)
        # ... same idea for encoder_outputs and decoder_outputs
        if self.rnn.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
            else:
                inflated_encoder_outputs = self._inflate(encoder_outputs, self.k)
        else:
            inflated_encoder_outputs = None

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = torch.Tensor(b*self.k, 1)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.LongTensor([i*self.k for i in range(0, b)]), 0.0)
        sequence_scores = Variable(sequence_scores)

        # Initialize the input vector
        input_var = Variable(torch.transpose(torch.LongTensor([[self.SOS]*b*self.k]), 0, 1))

        # Store decisions for backtracking
        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()

        for _ in range(0, self.rnn.max_length):

            # Run the RNN one step forward
            log_softmax_output, hidden, _ = self.rnn.forward_step(input_var, hidden, inflated_encoder_outputs, function=function)

            # If doing local backprop (e.g. supervised training), retain the output layer
            if retain_output_probs:
                stored_outputs.append(log_softmax_output)

            # To get the full sequence scores for the new candidates, add the local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = self._inflate(sequence_scores, self.V)
            sequence_scores += log_softmax_output
            scores, candidates = sequence_scores.view(b, -1).topk(self.k, dim=1)

            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            input_var = (candidates % self.V).view(b * self.k, 1)
            sequence_scores = scores.view(b * self.k, 1)

            # Update fields for next timestep
            predecessors = (candidates / self.V + self.pos_index.expand_as(candidates)).view(b*self.k, 1)
            hidden = hidden.index_select(1, predecessors.squeeze())

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = input_var.data.eq(self.EOS)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)
            stored_hidden.append(hidden)

        # Do backtracking to return the optimal values
        output, h_t, h_n, s, l, p = self._backtrack(stored_outputs, stored_hidden,
                                                 stored_predecessors, stored_emitted_symbols, stored_scores, b, h)

        # Build return objects
        decoder_outputs = [step[:, 0, :] for step in output]
        decoder_hidden = h_n[:, :, 0, :]
        metadata = {}
        metadata['inputs'] = inputs
        metadata['output'] = output
        metadata['h_t'] = h_t
        metadata['score'] = s
        metadata['length'] = l
        metadata['sequence'] = p
        return decoder_outputs, decoder_hidden, metadata

    def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, b, hidden_size):
        """Backtracks over batch to generate optimal k-sequences.

        Args:
            nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
            nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network
            predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
            symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            b: Size of the batch
            hidden_size: Size of the hidden state

        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]

            h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]

            h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.

            score [batch, k]: A list containing the final scores for all top-k sequences

            length [batch, k]: A list specifying the length of each sequence in the top-k candidates

            p (batch, k, sequence_len): A Tensor containing predicted sequence
        """

        # initialize return variables given different types
        output = list()
        h_t = list()
        p = list()
        h_n = torch.zeros(nw_hidden[0].size())  # Placeholder for last hidden state of top-k sequences.
                                                # If a (top-k) sequence ends early in decoding, `h_n` contains
                                                # its hidden state when it sees EOS.  Otherwise, `h_n` contains
                                                # the last hidden state of decoding.
        l = [[self.rnn.max_length] * self.k for _ in range(b)]  # Placeholder for lengths of top-k sequences
                                                                # Similar to `h_n`

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(b, self.k).topk(self.k)
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * b   # the number of EOS found
                                    # in the backward loop below for each batch

        t = self.rnn.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.k)
        while t >= 0:
            # Re-order the variables with the back pointer
            current_output = nw_output[t].index_select(0, t_predecessors)
            current_hidden = nw_hidden[t].index_select(1, t_predecessors)
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.squeeze(1).eq(self.EOS).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = idx[0] / self.k
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.k + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_output[res_idx, :] = nw_output[t][idx[0], :]
                    current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :]
                    h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data
                    current_symbol[res_idx, :] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            output.append(current_output)
            h_t.append(current_hidden)
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.k)
        for b_idx in range(b):
            l[b_idx] = [l[b_idx][k_idx.data[0]] for k_idx in re_sorted_idx[b_idx,:]]

        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        output = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(output)]
        p = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(p)]
        h_t = [step.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for step in reversed(h_t)]
        h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size)
        s = s.data

        if self.k == 1:
            l = [_l[0] for _l in l]

        return output, h_t, h_n, s, l, p

    def _mask_symbol_scores(self, score, idx, masking_score=-float('inf')):
            score[idx] = masking_score

    def _mask(self, tensor, idx, dim=0, masking_score=-float('inf')):
        if len(idx.size()) > 0:
            indices = idx[:, 0]
            tensor.index_fill_(dim, indices, masking_score)

    def _inflate(self, tensor, times):
        """
        Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)

        Args:
            tensor: A :class:`Tensor` to inflate
            times: number of repetitions
            dimension: axis for inflation (default=0)

        Returns:
            A :class:`Tensor`

        Examples::
            >> a = torch.LongTensor([[1, 2], [3, 4]])
            >> a
            1   2
            3   4
            [torch.LongTensor of size 2x2]
            >> decoder = TopKDecoder(nn.RNN(10, 20, 2), 3)
            >> b = decoder._inflate(a, 1, dimension=1)
            >> b
            1   1   2   2
            3   3   4   4
            [torch.LongTensor of size 2x4]
            >> c = decoder._inflate(a, 1, dimension=0)
            >> c
            1   2
            1   2
            3   4
            3   4
            [torch.LongTensor of size 4x2]

        """
        tensor_dim = len(tensor.size())
        if tensor_dim is 3:
            b = tensor.size(1)
            return tensor.repeat(1, 1, times).view(tensor.size(0), b * times, -1)
        elif tensor_dim is 2:
            return tensor.repeat(1, times)
        elif tensor_dim is 1:
            b = tensor.size(0)
            return tensor.repeat(times).view(b, -1)
        else:
            raise ValueError("Tensor can be of 1D, 2D or 3D only. This one is {}D.".format(tensor_dim))

