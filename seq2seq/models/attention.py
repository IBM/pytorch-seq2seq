import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    r"""
    Applies an attention Mechansims on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i - max_i x_i) / sum_j exp(x_j - max_i x_i) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, seq_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, seq_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, seq_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        # (batch, len, dim) * (batch, dim, 1) -> (batch, len)
        attn = torch.bmm(context, output.unsqueeze(2)).squeeze(2)
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn)

        # (batch, 1, len) * (batch, len, dim) -> (batch, dim)
        mix = torch.bmm(attn.unsqueeze(1), context).squeeze(1)

        combined = torch.cat((mix, output), 1)
        output = F.tanh(self.linear_out(combined))

        return output, attn
