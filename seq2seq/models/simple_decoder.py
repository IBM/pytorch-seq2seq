import torch.nn as nn
import torch.nn.functional as F

class SimpleDecoder(nn.Module):
    """ Simple Decoder Model """

    def __init__(self, hidden_size, output_size):
        super(SimpleDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, context, *args):
        batch_size, de_len = context.size(0), context.size(1)
        logits = self.linear(context.view(-1, self.hidden_size))
        softmax = F.softmax(logits, dim=-1).view(batch_size, de_len, self.output_size)
        symbols = softmax.topk(1, dim=2)[1]
        return softmax, symbols