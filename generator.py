import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init
from chsmm_without_src import HSMM
from torch.autograd import Variable
import random
import generate


class Generator(nn.Module):

    def __init__(self, hsmm_net, hid_size, vocab, dropout=0.5):
        super(Generator, self).__init__()
        self.hsmm_net = hsmm_net
        for param in self.hsmm_net.parameters():
            param.requires_grad = False
        for param in self.hsmm_net.seg_rnn.parameters():
            param.requires_grad = True
        hsmm_net.h0_lin.required_grad = True
        hsmm_net.state_out_gates.required_grad = True
        hsmm_net.state_out_biases.required_grad = True


        self.hid_size = hid_size
        self.lut = hsmm_net.lut
        self.emb_size = hsmm_net.emb_size
        self.K = hsmm_net.K
        self.L = hsmm_net.L
        self.stack_rnn = nn.LSTM(self.hsmm_net.hid_size, self.hid_size, 1)
        self.encode_rnn = nn.LSTM(self.emb_size, self.hid_size, 1, dropout=dropout, bidirectional=True)
        self.encode_redu = nn.Linear(self.hid_size * 2, self.hid_size)
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.decoder = nn.Linear(2 * self.hid_size, self.vocab_size)

        self.attn_W = nn.Linear(self.hid_size, self.hid_size, bias=False)
        self.attn_U = nn.Linear(self.hid_size, self.hid_size)
        self.V = nn.Linear(self.hid_size, 1, bias=False)

        # debug
        self.debug_rnn = nn.LSTM(self.hsmm_net.emb_size, self.hid_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # params = [ self.V]
        # for param in params:
        #     param.data.uniform_(-initrange, initrange)
        lins = [self.attn_W, self.attn_U, self.decoder, self.encode_redu, self.V]
        for lin in lins:
            lin.weight.data.uniform_(-initrange, initrange)
            if lin.bias is not None:
                lin.bias.data.zero_()

    def forward(self, src, inps, template):
        '''
        get generated responses.
        Args:
            src: bsz x src_len
            inps: bsz x response_len
            template: bsz x response_len
        Returns:
            response_samples: bsz x response_len
        '''
        # tpl = self.template_sample(templates)
        bsz, inp_len = inps.size()
        src_enc, src_sent_enc = self.encode(src)
        src_mask = self.make_src_masks(src)

        # split the inps into segs
        # begin = [sum(seg_lens[:i]) for i, _ in enumerate(seg_lens)]
        # end = [b + l for b, l in zip(begin, seg_lens)]

        inits = self.hsmm_net.h0_lin.unsqueeze(0)
        h1, c1 = torch.tanh(inits[:, :self.hsmm_net.hid_size]), inits[:, self.hsmm_net.hid_size:]
        h1 = h1.expand(1, bsz, self.hsmm_net.hid_size).contiguous()
        c1 = c1.expand(1, bsz, self.hsmm_net.hid_size).contiguous()

        h2, c2 = torch.tanh(src_enc), src_enc  # bsz x hid_size
        h2, c2 = h2.unsqueeze(0).contiguous(), c2.unsqueeze(0).contiguous()  # 1 x bsz x hid_size

        start_inp = self.hsmm_net.start_emb
        inps_emb = self.hsmm_net.lut(inps)  # bsz x inp_len x emb_size
        inps_emb = torch.cat([start_inp.repeat(bsz, 1, 1), inps_emb], 1)  # bsz x inp_len+1 x emb_size

        vocab_dist = Variable(torch.zeros(inp_len, bsz, self.vocab_size)).cuda()
        for i in range(inp_len):
            inp_emb = inps_emb[:, i, :].unsqueeze(0)  # 1 x bsz x emb_size
            ss = template[:, i]  # bsz
            wlps_k, h1, c1, h2, c2 = self.get_next_word_dist(inp_emb, ss, h1, c1, h2, c2, src_sent_enc, src_mask)
            # todo: consider h1, c1 between segment.
            vocab_dist[i] = wlps_k

        vocab_dist = F.log_softmax(vocab_dist, 2)  # inp_len x bsz x V
        return vocab_dist, h1, c1, h2, c2

    def encode(self, src, mask=None):
        '''
        Args:
            src: bsz x src_len
            avgmask: bsz x src_len, with 1s for pad and 0s for rest
        Returns:
            bsz x hidden_dim, bsz x src_len x hidden_dim
        '''
        bsz, src_len = src.size()
        embs = self.lut(src)  # bsz x src_len x emb_size
        output, (hn, cn) = self.encode_rnn(embs.transpose(0, 1).contiguous())
        src_enc = self.encode_redu(
            hn.transpose(0, 1).contiguous().view(bsz, self.encode_rnn.num_layers, -1)[:, -1, :])  # bsz x hidden_dim
        src_sen_enc = self.encode_redu(output.transpose(0, 1))  # bsz x src_len x hidden_dim
        assert src_enc.size() == (bsz, self.hid_size)
        assert src_sen_enc.size() == (bsz, src_len, self.hid_size)
        return src_enc, src_sen_enc

    def make_src_masks(self, src):
        """
        :param src: bsz x src_seq_len
        return bsz x src_seq_len, padding part are neginf, remaing are 0.
        """
        neginf = -1e38
        pad_idx = self.hsmm_net.pad_idx
        srcmask = src.eq(pad_idx).float() * neginf  # bsz x src_sent_len
        return srcmask

    def get_next_word_dist(self, inp_emb, ss, h1, c1, h2, c2, src_sent_enc, src_mask):
        """
        get word dists for inp_emb.
        Args:
            inp_emb: 1 x bsz x emb_size
            state_emb: inp_len x bsz x emb_size*2
            ss: bsz
            h1: 1 x bsz x rnn_size  note: 1 is num_layers*direction
            c1: 1 x bsz x rnn_size
            h2: 1 x bsz x rnn_size
            c2: 1 x bsz x rnn_size
            src_sent_enc: bsz x src_len x rnn_size
            src_mask: bsz x src_len    the padding values are -inf
        Returns:
            wlps_k: 1 x bsz x V+1
            h1: 1 x bsz x rnn_size
            c1: 1 x bsz x rnn_size
            h2: 1 x bsz x rnn_size
            c2: 1 x bsz x rnn_size
        """
        _, bsz, _ = inp_emb.size()
        hsmm_state_embs = self.hsmm_net.state_embs.index_select(0, ss).squeeze(1).transpose(0,
                                                                                            1)  # 1 x bsz x hsmm_emb_size
        cond_start_inp = torch.cat([inp_emb, hsmm_state_embs], 2)  ## 1 x bsz x emb_size*3
        hid, (h1, c1) = self.hsmm_net.seg_rnn(cond_start_inp, (h1, c1))  # 1 x bsz x rnn_size

        hsmm_gates = self.hsmm_net.state_out_gates.index_select(0, ss).squeeze(1).transpose(0, 1)  # 1 x bsz x rnn_size
        hsmm_bias = self.hsmm_net.state_out_biases.index_select(0, ss).squeeze(1).transpose(0, 1)  # 1 x bsz x rnn_size
        states = hsmm_gates * hid + hsmm_bias

        states, (h2, c2) = self.stack_rnn(states, (h2, c2))  # 1 x bsz x rnn_size
        # states, (h2, c2) = self.debug_rnn(inp_emb, (h2, c2))
        # bahdanau attention
        attn = torch.tanh(self.attn_W(src_sent_enc) + self.attn_U(states.transpose(0, 1)))  # bsz x src_len x rnn_size
        ascores = self.V(attn).squeeze(2)  # bsz x src_len
        ascores += src_mask
        aprobs = torch.softmax(ascores, 1)  # bsz x src_len
        ctx = torch.bmm(aprobs.unsqueeze(1), src_sent_enc)  # bsz x 1 x rnn_size

        # dot attention
        # ascores = torch.bmm(states.transpose(0, 1).contiguous().view(bsz, 1, self.hid_size),
        #                     src_sent_enc.transpose(1, 2))  # bsz x inp_len x src_len
        # ascores += src_mask.unsqueeze(1)  # bsz x 1 x src_len
        # aprobs = F.softmax(ascores, dim=2)
        # ctx = torch.bmm(aprobs, src_sent_enc)  # bsz x 1 x rnn_size

        cat_ctx = torch.cat([states, ctx.transpose(0, 1)], 2)  # 1 x bsz x rnn_size*2

        wlps_k = self.decoder(cat_ctx)  # 1 x bsz x V
        return wlps_k, h1, c1, h2, c2

    def template_sample(self, templates, limit=None):
        '''
        sample a template for current batch.
        templates - templates list
        limit - the limit length
        Returns:
            a sampled template with limit length
        '''
        # init_lps, trans_lps = self.hsmm_net.ini
        _, len_lps = self.hsmm_net.len_logprobs()
        while True:
            tpl = templates[random.randint(0, len(templates) - 1)]
            seg_lens = []
            for ss in tpl:
                seg_lens.append(max(torch.multinomial(len_lps[ss], 1).item() + 1, 3))
            if not limit:
                break
            if limit and sum(seg_lens) <= limit:
                seg_lens[-1] += limit - sum(seg_lens)
                break

        final_tpl = []
        for s, l in zip(tpl, seg_lens):
            final_tpl += [s] * l
        return torch.LongTensor(final_tpl)

    def sample_one(self, src, inps, templt):
        """
        src - bsz x src_len
        inps - bsz x inp_len
        templt - bsz x response_len   (response_len > inp_len)
        returns:
            response_samples - bsz x template_len
        """
        bsz, _ = src.size()
        bsz, response_len = templt.size()
        inp_len = 0 if inps is None else inps.size(1)

        i2w, w2i = self.vocab.idx2word, self.vocab.word2idx
        unk_idx, eos_idx, pad_idx = w2i["<unk>"], w2i["<eos>"], w2i["<pad>"]

        src_enc, src_sent_enc = self.encode(src)
        src_mask = self.make_src_masks(src)

        rep_sample = torch.zeros(bsz, response_len).type(torch.LongTensor).cuda()
        if inp_len:
            bsz, inp_len = inps.size()
            inp_template = templt[:, :inp_len]
            _, h1, c1, h2, c2 = self.forward(src, inps, inp_template)
            rep_sample[:, :inp_len] = inps
            inp_emb = self.lut(inps[:, inp_len - 1]).unsqueeze(0)  # 1 x bsz x emb_size
        else:
            inits = self.hsmm_net.h0_lin.unsqueeze(0)
            h1, c1 = torch.tanh(inits[:, :self.hsmm_net.hid_size]), inits[:, self.hsmm_net.hid_size:]
            h1 = h1.expand(1, bsz, self.hsmm_net.hid_size).contiguous()
            c1 = c1.expand(1, bsz, self.hsmm_net.hid_size).contiguous()

            h2, c2 = torch.tanh(src_enc), src_enc  # bsz x hid_size
            h2, c2 = h2.unsqueeze(0).contiguous(), c2.unsqueeze(0).contiguous()  # 1 x bsz x hid_size

            start_inp = self.hsmm_net.start_emb  # 1 x 1 x emb_size
            inp_emb = start_inp.repeat(1, bsz, 1)  # 1 x bsz x emb_size

        for i in range(inp_len, response_len):
            ss = templt[:, i]
            wlps, h1, c1, h2, c2 = self.get_next_word_dist(inp_emb, ss, h1, c1, h2, c2, src_sent_enc, src_mask)
            wlps = torch.softmax(wlps[0], 1)  # bsz x V
            wlps[:, unk_idx].zero_()  # remove the unk id
            if i < 5:
                wlps[:, eos_idx].zero_()

            #constraint to top50 words
            probs, topk_ids = torch.topk(wlps,50)
            nor_probs = probs / torch.sum(probs, 1).unsqueeze(1) #bsz x 50
            sids = torch.multinomial(nor_probs, 1)    #bsz x 1
            sample_ids = torch.gather(topk_ids,1, sids).squeeze(1)  #bsz

            # sample_ids = torch.multinomial(wlps, 1).squeeze(1)  # bsz
            # # _, sample_ids = torch.topk(wlps, 1)
            # # sample_ids = sample_ids.squeeze(1)
            rep_sample[:, i] = sample_ids
            inp_emb = self.lut(sample_ids).unsqueeze(0)

        # post-process: The ids after first <eos> are set to pad ids.
        for i in range(bsz):
            rep = rep_sample[i]
            index = np.where(rep.cpu().numpy() == eos_idx)[0]
            if len(index) >= 1:
                rep_sample[i][index[0] + 1:] = pad_idx

        # src = [[i2w[_] for _ in src[i]] for i in range(bsz)]
        # rep = [[i2w[_] for _ in rep_sample[i]] for i in range(bsz)]

        return rep_sample

    def beam_sample(self, src, templt, beam_size=5):
        """
            src - bsz x src_len
            inps - bsz x inp_len
            templt - bsz x response_len   (response_len > inp_len)
            returns:
                response_samples - bsz x template_len
        """
        i2w, w2i = self.vocab.idx2word, self.vocab.word2idx
        unk_idx, eos_idx, pad_idx = w2i["<unk>"], w2i["<eos>"], w2i["<pad>"]

        bsz, tpl_len = templt.size()
        rep_sample = torch.full((bsz, tpl_len), pad_idx, dtype=torch.long).cuda()
        for i in range(bsz):
            rep, _ = generate.beam_search(self, self.vocab, src.narrow(0, i, 1), beam_size, templt[i])
            rep_sample[i][:len(rep)] = torch.stack(rep, 0)

        # src = [[i2w[_] for _ in s] for s in src]
        # res = [[i2w[_] for _ in r] for r in rep_sample]

        return rep_sample

    def batch_nll_loss(self, src, inps, templates):
        '''
        return nll loss for predicting target sequence.
        Args:
            src: bsz x src_len
            inps: bsz x inp_len
            templates: bsz x inp_len
        Returns:
            the nll loss.
        '''
        bsz, inp_len = inps.size()
        loss_fn = nn.NLLLoss()
        vocab_dist, _, _, _, _ = self.forward(src, inps, templates)  # inp_len x bsz x V
        loss = 0
        for i in range(inp_len):
            dist = vocab_dist[i]
            loss += loss_fn(dist, inps[:, i])
        return loss

    def batch_reward_loss(self, query, response, templates, rewards):
        '''
        Args:
            query: bsz x src_len
            response: bsz x inp_len
            templates: bsz x inp_len
            rewards: bsz x inp_len
            mask: bsz x inp_len

        Returns:
            loss
        '''
        pad_idx = self.vocab.word2idx['<pad>']
        loss_fn = nn.NLLLoss(ignore_index=pad_idx)

        bsz, inp_len = response.size()
        vocab_dist, _, _, _, _ = self.forward(query, response, templates)  # inp_len x bsz x V
        reward_dist = vocab_dist * rewards.t().unsqueeze(2).expand_as(vocab_dist)
        loss = 0
        for i in range(inp_len):
            dist = reward_dist[i]
            loss += loss_fn(dist, response[:, i])  # ignore the pad idx in response
        return loss

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
