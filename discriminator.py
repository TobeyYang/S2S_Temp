import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb
from textcnn import TextCNN


class Discriminator(nn.Module):

    def __init__(self, vocab_size, emb_dim, filter_num, filter_sizes, dropout=0.0):
        super(Discriminator, self).__init__()
        self.query_cnn = TextCNN(emb_dim, filter_num, filter_sizes)
        self.response_cnn = TextCNN(emb_dim, filter_num, filter_sizes)
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        #
        self.judger = nn.Sequential(
            nn.Linear(2 * filter_num * len(filter_sizes), 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        # self.query_h0 = nn.Parameter(torch.zeros(1, 1, 300))
        # self.query_c0 = nn.Parameter(torch.zeros(1, 1, 300))
        # self.response_h0 = nn.Parameter(torch.zeros(1, 1, 300))
        # self.response_c0 = nn.Parameter(torch.zeros(1, 1, 300))
        #
        # self.query_rnn = torch.nn.LSTM(emb_dim, 300, 1)
        # self.response_rnn = torch.nn.LSTM(emb_dim, 300, 1)
        # self.judger = nn.Sequential(
        #     nn.Linear(2 * 300, 128),
        #     nn.ReLU(),
        #     self.dropout,
        #     nn.Linear(128, 2),
        #     nn.Softmax(dim=1)
        #
        # )

    def forward(self, query, response):
        '''

        Args:
            query:  bsz x query_len
            response:  bsz x response_len

        Returns:

        '''
        bsz = query.size(0)
        query_emb = self.embeddings(query)  # bsz x query_len x emb_size
        response_emb = self.embeddings(response)  # bsz x response_len x emb_size
        query_features = self.query_cnn(query_emb)  # [B, T, D] -> [B, all_features]
        response_features = self.response_cnn(response_emb)

        # h0 = self.query_h0.expand(1, bsz, 300).contiguous()
        # c0 = self.query_c0.expand(1, bsz, 300).contiguous()
        # _, (h_n, c_n) = self.query_rnn(query_emb.transpose(0, 1), (h0, c0))
        # query_features = h_n.squeeze(0)  # bsz x 300
        #
        # h0 = self.response_h0.expand(1, bsz, 300).contiguous()
        # c0 = self.response_c0.expand(1, bsz, 300).contiguous()
        # _, (h_n, c_n) = self.response_rnn(response_emb.transpose(0, 1), (h0, c0))
        # response_features = h_n.squeeze(0)  # bsz x 300

        inputs = torch.cat((query_features, response_features), 1)
        prob = self.judger(inputs)[:, 1]
        return prob

    def batchClassify(self, query, response):
        '''
        Args:
            query:  bsz x query_len
            response:   bsz x response_len
        Returns:
            out: bsz
        '''
        out = self.forward(query, response)
        return out

    def batchBCEloss(self, query, response, target):
        '''
        Returns Binary Cross Entropy Loss for discriminator.
        Args:
            query: bsz x query_len
            response:   bsz x response_len
            target: bsz (binary 0/1)

        Returns:
        '''
        loss_fn = nn.BCELoss()
        out = self.forward(query, response)
        return loss_fn(out, target)

    def validate(self, valid_set):
        acc = 0
        total = 0
        for i in range(len(valid_set)):
            src, rep, target = valid_set[i]
            bsz, _ = src.size()
            out = self.batchClassify(src, rep)
            acc += torch.sum((out > 0.5) == (target > 0.5)).data.item()
            total += bsz

        return acc / total

    def save_model(self, save_path, **kwargs):
        kwargs['state_dict'] = self.state_dict()
        torch.save(kwargs, save_path)

    def load_model(self, save_path):
        saved_stuff = torch.load(save_path)
        # remove the 'module' prefix
        clean_state = {}
        saved_state = saved_stuff["state_dict"]
        for k, v in saved_state.items():
            nk = k[7:] if k.startswith('module.') else k
            clean_state[nk] = v
        # self.load_state_dict(saved_stuff['state_dict'])
        self.load_state_dict(clean_state)
