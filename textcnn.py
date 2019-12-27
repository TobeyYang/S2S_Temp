# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TextCNN(nn.Module):
    """TextCNN"""

    def __init__(self, input_dim, filter_num, filter_sizes):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (width, input_dim)) for width in filter_sizes])

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, embedded_inputs):
        x = embedded_inputs.unsqueeze(1)  # (N, 1, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(B, filter_num), ...]*len(Ks)
        return torch.cat(x, 1)  # [B, all_of_features]


if __name__ == '__main__':
    vocab_size = 10
    embed_dim = 6
    embeddings = nn.Embedding(vocab_size, embed_dim)
    inputs = Variable(torch.LongTensor([[4, 5, 5, 3, 3, 0, 0, 0], [6, 3, 0, 0, 0, 0, 0, 0]]))
    embedded_inputs = embeddings(inputs)
    x = embedded_inputs.unsqueeze(1)
    print(x.size())

    filter_num = 5
    filter_size = 3
    conv = nn.Conv2d(1, filter_num,
                     (filter_size, embed_dim))  # [B, filter_num, out_width, 1] -> [B, filter_num, out_width]
    conv_res = F.relu(conv(x)).squeeze(3)
    print(conv_res.shape)
    pool_res = F.max_pool1d(conv_res, conv_res.size(2))
    print(pool_res.shape)
