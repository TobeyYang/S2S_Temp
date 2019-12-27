import sys
import os
import math
import random
import numpy as np
import argparse
from collections import defaultdict, Counter
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import datasets
from utils import logsumexp1, make_bwd_constr_idxs, backtrace
import infc


class HSMM(nn.Module):
    """
    standard hsmm
    """

    def __init__(self, wordtypes, gentypes, opt):
        super(HSMM, self).__init__()
        self.K = opt.K
        self.Kmul = opt.Kmul

        self.L = opt.L
        self.A_dim = opt.A_dim
        self.unif_lenps = opt.unif_lenps
        self.A_from = nn.Parameter(torch.Tensor(opt.K * opt.Kmul, opt.A_dim))
        self.A_to = nn.Parameter(torch.Tensor(opt.A_dim, opt.K * opt.Kmul))

        hsmm_emb_size = opt.emb_size * 2
        rnninsz = opt.emb_size + hsmm_emb_size

        if self.unif_lenps:
            self.len_scores = nn.Parameter(torch.ones(1, opt.L))
            self.len_scores.requires_grad = False
        else:
            self.len_decoder = nn.Linear(hsmm_emb_size, opt.L)

        self.yes_self_trans = opt.yes_self_trans
        if not self.yes_self_trans:
            selfmask = torch.Tensor(opt.K * opt.Kmul).fill_(-float("inf"))
            self.register_buffer('selfmask', Variable(torch.diag(selfmask), requires_grad=False))

        self.emb_size, self.layers, self.hid_size = opt.emb_size, opt.layers, opt.hid_size
        self.pad_idx = opt.pad_idx
        self.lut = nn.Embedding(wordtypes, opt.emb_size, padding_idx=opt.pad_idx)

        # category_mlp
        self.cat_mlp = nn.Sequential(nn.Linear(hsmm_emb_size, self.hid_size),
                                     nn.ReLU(),
                                     nn.Linear(self.hid_size, 3))

        self.start_emb = nn.Parameter(torch.Tensor(1, 1, self.emb_size))
        self.pad_emb = nn.Parameter(torch.zeros(1, 1, self.emb_size))

        self.seg_rnn = nn.LSTM(rnninsz, opt.hid_size, opt.layers, dropout=opt.dropout)

        self.state_embs = nn.Parameter(
            torch.Tensor(opt.K, 1, 1, hsmm_emb_size))  # hint: the hsmm dim is twice larger than rnn
        self.trans_weights = nn.Parameter(torch.Tensor(hsmm_emb_size, hsmm_emb_size))
        self.trans_bias = nn.Parameter(torch.Tensor(opt.K, opt.K))

        # self.encode_rnn = nn.LSTM(opt.emb_size, opt.hid_size, opt.layers, dropout=opt.dropout, bidirectional=True)
        # self.encode_redu = nn.Linear(opt.hid_size * 2, opt.hid_size)

        # self.h0_lin = nn.Linear(opt.hid_size, 2 * opt.hid_size)
        self.h0_lin = nn.Parameter(torch.zeros(2 * opt.hid_size))
        # self.state_att_gates = nn.Parameter(torch.Tensor(opt.K, 1, 1, opt.hid_size))
        # self.state_att_biases = nn.Parameter(torch.Tensor(opt.K, 1, 1, opt.hid_size))

        # out_hid_sz = opt.hid_size + opt.emb_size
        out_hid_sz = opt.hid_size
        self.state_out_gates = nn.Parameter(torch.Tensor(opt.K, 1, 1, out_hid_sz))
        self.state_out_biases = nn.Parameter(torch.Tensor(opt.K, 1, 1, out_hid_sz))
        # add one more output word for eop
        self.decoder = nn.Linear(out_hid_sz, gentypes + 1)
        self.eop_idx = gentypes
        self.attn_lin1 = nn.Linear(opt.hid_size, opt.emb_size)
        self.linear_out = nn.Linear(opt.hid_size + opt.emb_size, opt.hid_size)

        self.drop = nn.Dropout(opt.dropout)
        self.emb_drop = opt.emb_drop
        self.initrange = opt.initrange
        self.lsm = nn.LogSoftmax(dim=1)
        self.zeros = torch.zeros(1, 1)
        if opt.cuda:
            self.zeros = self.zeros.cuda()

        # src encoder stuff
        self.src_bias = nn.Parameter(torch.Tensor(1, opt.emb_size))
        self.uniq_bias = nn.Parameter(torch.Tensor(1, opt.emb_size))

        self.init_lin = nn.Linear(opt.emb_size, opt.K * opt.Kmul)
        self.init_trans = nn.Parameter(torch.ones(1, opt.K * opt.Kmul))

        if opt.unif_trans:
            self.init_trans.required_grad = False

        self.cond_A_dim = opt.cond_A_dim
        self.cond_trans_lin = nn.Linear(opt.emb_size, opt.K * opt.Kmul * opt.cond_A_dim * 2)
        self.init_weights()

    def init_weights(self):
        """
        (re)init weights
        """
        initrange = self.initrange
        self.lut.weight.data.uniform_(-initrange, initrange)
        self.lut.weight.data[self.pad_idx].zero_()
        params = [self.src_bias, self.state_out_gates, self.state_out_biases, self.start_emb, self.uniq_bias]
        params.append(self.state_embs)

        for param in params:
            param.data.uniform_(-initrange, initrange)

        rnns = [self.seg_rnn]
        for rnn in rnns:
            for thing in rnn.parameters():
                thing.data.uniform_(-initrange, initrange)

        lins = [self.init_lin, self.decoder, self.attn_lin1, self.linear_out, self.cond_trans_lin]

        if not self.unif_lenps:
            lins.append(self.len_decoder)
        for lin in lins:
            lin.weight.data.uniform_(-initrange, initrange)
            if lin.bias is not None:
                lin.bias.data.zero_()

        def init_seq(m):
            if type(m) == nn.Linear:
                m.weight.data.uniform_(-initrange, initrange)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.cat_mlp.apply(init_seq)

    def trans_logprobs(self, bsz, seqlen):
        """
        Returns:
            1 x K tensor and seqlen-1 x bsz x K x K tensor of log probabilities,
                           where lps[i] is p(q_{i+1} | q_i)
        """
        K = self.K * self.Kmul
        state_embs = self.state_embs.squeeze()  # K x d
        tscores = torch.mm(torch.mm(state_embs, self.trans_weights), state_embs.t()) + self.trans_bias
        if not self.yes_self_trans:
            tscores = tscores + self.selfmask

        tscores = self.lsm(tscores)
        trans_lps = tscores.unsqueeze(0).expand(bsz, K, K)

        init_lps = self.lsm(self.init_trans).expand(bsz, K)
        return init_lps, trans_lps.view(1, bsz, K, K).expand(seqlen - 1, bsz, K, K)

    def len_logprobs(self):
        """
        Returns:
            [1xK tensor, 2 x K tensor, .., L-1 x K tensor, L x K tensor] of logprobs
        """
        K = self.K * self.Kmul
        if self.unif_lenps:
            len_scores = self.len_scores.expand(K, self.L)
        else:
            len_scores = self.len_decoder(self.state_embs.squeeze())  # K x L
        lplist = [Variable(len_scores.data.new(1, K).zero_())]  # p=1 log(p)=0
        for l in range(2, self.L + 1):
            lplist.append(self.lsm(len_scores.narrow(1, 0, l)).t())
        return lplist, len_scores

    def to_seg_embs(self, xemb):
        """
        xemb - bsz x seqlen x emb_size
        returns - L+1 x bsz*seqlen x emb_size,
           where [1 2 3 4]  becomes [<s> <s> <s> <s> <s> <s> <s> <s>]
                 [5 6 7 8]          [ 1   2   3   4   5   6   7   8 ]
                                    [ 2   3   4  <p>  6   7   8  <p>]
                                    [ 3   4  <p> <p>  7   8  <p> <p>]
        """
        bsz, seqlen, emb_size = xemb.size()
        newx = [self.start_emb.expand(bsz, seqlen, emb_size)]
        newx.append(xemb)
        for i in range(1, self.L):
            pad = self.pad_emb.expand(bsz, i, emb_size)
            rowi = torch.cat([xemb[:, i:], pad], 1)
            newx.append(rowi)
        # L+1 x bsz x seqlen x emb_size -> L+1 x bsz*seqlen x emb_size
        return torch.stack(newx).view(self.L + 1, -1, emb_size)

    def to_seg_hist(self, states):
        """
        states - bsz x seqlen+1 x rnn_size
        returns - L+1 x bsz*seqlen x emb_size,
           where [<b> 1 2 3 4]  becomes [<b>  1   2   3  <b>  5   6   7 ]
                 [<b> 5 6 7 8]          [ 1   2   3   4   5   6   7   8 ]
                                        [ 2   3   4  <p>  6   7   8  <p>]
                                        [ 3   4  <p> <p>  7   8  <p> <p>]
        """
        bsz, seqlenp1, rnn_size = states.size()
        newh = [states[:, :seqlenp1 - 1, :]]  # [bsz x seqlen x rnn_size]
        newh.append(states[:, 1:, :])
        for i in range(1, self.L):
            pad = self.pad_emb[:, :, :rnn_size].expand(bsz, i, rnn_size)
            rowi = torch.cat([states[:, i + 1:, :], pad], 1)
            newh.append(rowi)
        # L+1 x bsz x seqlen x rnn_size -> L+1 x bsz*seqlen x rnn_size
        return torch.stack(newh).view(self.L + 1, -1, rnn_size)

    def obs_logprobs(self, x, combotargs, vocab_masks):
        """
        args:
          x - seqlen x bsz
          combotargs - L x bsz*seqlen
          vocab_masks - 3 x gentypes
        returns:
          a L x seqlen x bsz x K tensor, where l'th row has prob of sequences of length l+1.
          specifically, obs_logprobs[:,t,i,k] gives p(x_t|k), p(x_{t:t+1}|k), ..., p(x_{t:t+l}|k).
          the infc code ignores the entries rows corresponding to x_{t:t+m} where t+m > T
        """
        seqlen, bsz = x.size()
        embs = self.lut(x)  # seqlen x bsz x emb_size
        inpembs = self.drop(embs) if self.emb_drop else embs

        # get L+1 x bsz*seqlen x emb_size segembs
        segembs = self.to_seg_embs(inpembs.transpose(0, 1))
        Lp1, bszsl, _ = segembs.size()

        layers, rnn_size = self.layers, self.hid_size

        # bsz x dim -> bsz x seqlen x dim -> bsz*seqlen x dim -> layers x bsz*seqlen x dim
        inits = self.h0_lin.unsqueeze(0).expand(bsz, 2 * rnn_size)  # bsz x 2*dim
        h0, c0 = inits[:, :rnn_size], inits[:, rnn_size:]  # (bsz x dim, bsz x dim)
        h0 = F.tanh(h0).unsqueeze(1).expand(bsz, seqlen, rnn_size).contiguous().view(
            -1, rnn_size).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()
        c0 = c0.unsqueeze(1).expand(bsz, seqlen, rnn_size).contiguous().view(
            -1, rnn_size).unsqueeze(0).expand(layers, -1, rnn_size).contiguous()    # [1,bsz*seqlen,dim]

        # easiest to just loop over K
        state_emb_sz = self.state_embs.size(3)
        seg_lls = []
        for k in range(self.K):

            condembs = torch.cat([segembs, self.state_embs[k].expand(Lp1, bszsl, state_emb_sz)], 2)
            states, _ = self.seg_rnn(condembs, (h0, c0))  # L+1 x bsz*seqlen x rnn_size

            if self.drop.p > 0:
                states = self.drop(states)

            states = self.state_out_gates[k].expand_as(states) * states + self.state_out_biases[k].expand_as(states)

            out_hid_sz = rnn_size
            states_k = states  # L+1 x bsz*seqlen x out_hid_sz

            # vocab projection: weight by vocab mask and then normalize
            if args.no_mask:
                wlps_k = F.softmax(self.decoder(states_k.view(-1, out_hid_sz)), 1)
            else:
                cat_dist = F.softmax(self.cat_mlp(self.state_embs[k].squeeze()), 0)
                # print(f'k:{k}', cat_dist)
                vocab_mask = torch.sum(cat_dist.unsqueeze(1).expand_as(vocab_masks) * vocab_masks, 0)  # V
                # vocab_mask, _ = torch.max(cat_dist.unsqueeze(1).expand_as(vocab_masks) * vocab_masks, 0)  # V
                vocab_mask = torch.cat([vocab_mask, torch.Tensor([1]).cuda()], 0)  # V+1

                wlps_k = F.softmax(self.decoder(states_k.view(-1, out_hid_sz)), 1) * vocab_mask  # L+1*bsz*seqlen x V+1
                wlps_k = wlps_k / wlps_k.sum(1, keepdim=True)

            # concatenate on dummy column for when only a single answer...
            wlps_k = torch.cat([wlps_k, self.zeros.expand(wlps_k.size(0), 1)], 1)
            # L+1*bsz*sl x (V+1)

            # get scores for predicted next-words (but not for last words in each segment as usual)
            psk = wlps_k.narrow(0, 0, self.L * bszsl).gather(1, combotargs.view(self.L * bszsl, -1))
            lls_k = psk.sum(1).log()  # L*bsz*seqlen   #todo: log the dummy is right?

            # sum up log probs of words in each segment
            seglls_k = lls_k.view(self.L, -1).cumsum(0)  # L x bsz*seqlen
            # need to add end-of-phrase prob too
            eop_lps = wlps_k.narrow(0, bszsl, self.L * bszsl)[:, self.eop_idx]  # L*bsz*seqlen
            seglls_k = seglls_k + eop_lps.log().view(self.L, -1)  # L x bsz*seqlen
            seg_lls.append(seglls_k)

        #  K x L x bsz x seqlen -> seqlen x L x bsz x K -> L x seqlen x bsz x K
        obslps = torch.stack(seg_lls).view(self.K, self.L, bsz, -1).transpose(
            0, 3).transpose(0, 1)
        if self.Kmul > 1:
            obslps = obslps.repeat(1, 1, 1, self.Kmul)
        return obslps

    def get_next_word_dist(self, hid, k):
        """
        get current word dist.
        Args:
            hid: 1 x bsz x rnn_size
            k: the number of k-th state.
        Returns:
        """
        _, bsz, rnn_size = hid.size()
        states = self.state_out_gates[k].expand_as(hid) * hid + self.state_out_biases[k].expand_as(hid)
        states_k = states  # 1 x bsz x rnn_size
        # if args.no_mask:
        #     wlps_k = F.softmax(self.decoder(states_k.view(-1, rnn_size)), 1)
        # else:
        #     wlps_k = F.softmax(self.decoder(states_k.view(-1, rnn_size)), 1) * vocab_mask  # L+1*bsz*seqlen x V+1
        #     wlps_k = wlps_k / wlps_k.sum(1, keepdim=True)
        wlps_k = F.softmax(self.decoder(states_k.view(-1, rnn_size)), 1)  # bsz x V+1
        return wlps_k.data

    def temp_bs(self, ss, start_inp, exh0, exc0, len_lps, row2tblent, row2feats, beam_size, final_state=False):
        """
        ss - discrete state index
        exh0 - layers x 1 x rnn_size
        exc0 - layers x 1 x rnn_size
        start_inp - 1 x 1 x emb_size
        len_lps - K x L, log normalized
        """
        rul_ss = ss % self.K
        i2w = corpus.dictionary.idx2word
        w2i = corpus.dictionary.word2idx
        unk_idx, eos_idx, pad_idx = w2i["<unk>"], w2i["<eos>"], w2i["<pad>"]
        state_emb_sz = self.state_embs.size(3)

        cond_start_inp = torch.cat([start_inp, self.state_embs[rul_ss]], 2)  # 1 x 1 x cat_size
        hid, (hc, cc) = self.seg_rnn(cond_start_inp, (exh0, exc0))

        curr_hyps = [(None, None)]
        best_wscore, best_lscore = None, None  # so we can truly average over words etc later
        best_hyp, best_hyp_score = None, -float("inf")
        curr_scores = torch.zeros(beam_size, 1)
        # N.B. we assume we have a single feature row for each timestep rather than avg
        # over them as at training time. probably better, but could conceivably average like
        # at training time.
        # inps = Variable(torch.LongTensor(K, 4), volatile=True)

        inps = torch.LongTensor(beam_size)
        for ell in range(self.L):
            wrd_dist = self.get_next_word_dist(hid, rul_ss).cpu()  # 1 x V+1
            # disallow unks
            wrd_dist[:, unk_idx].zero_()
            if not final_state:
                wrd_dist[:, eos_idx].zero_()
            # self.collapse_word_probs(row2tblent, wrd_dist, corpus)
            wrd_dist.log_()
            if ell > 0:  # add previous scores
                wrd_dist.add_(curr_scores.expand_as(wrd_dist))
            maxprobs, top2k = torch.topk(wrd_dist.view(-1), 2 * beam_size)
            cols = wrd_dist.size(1)
            # we'll break as soon as <eos> is at the top of the beam.
            # this ignores <eop> but whatever
            if top2k[0] == eos_idx:
                final_hyp = backtrace(curr_hyps[0])
                final_hyp.append(eos_idx)
                return final_hyp, maxprobs[0], len_lps[ss][ell]

            new_hyps, anc_hs, anc_cs = [], [], []
            for k in range(2 * beam_size):
                anc, wrd = top2k[k] / cols, top2k[k] % cols
                # check if any of the maxes are eop
                if wrd == self.eop_idx and ell > 0:
                    # add len score (and avg over num words incl eop i guess)
                    wlenscore = maxprobs[k] / (ell + 1) + len_lps[ss][ell - 1]
                    if wlenscore > best_hyp_score:
                        best_hyp_score = wlenscore
                        best_hyp = backtrace(curr_hyps[anc])
                        best_wscore, best_lscore = maxprobs[k], len_lps[ss][ell - 1]
                else:
                    curr_scores[len(new_hyps)][0] = maxprobs[k]
                    if wrd >= self.decoder.out_features:  # a copy
                        tblidx = wrd - self.decoder.out_features
                        inps.data[len(new_hyps)].copy_(row2feats[tblidx])
                    else:
                        inps.data[len(new_hyps)] = wrd if (
                                wrd < len(i2w) and wrd >= 4) else unk_idx  # the fist 4 are symbols.
                    new_hyps.append((wrd, curr_hyps[anc]))
                    anc_hs.append(hc.narrow(1, anc, 1))  # layers x 1 x rnn_size
                    anc_cs.append(cc.narrow(1, anc, 1))  # layers x 1 x rnn_size
                if len(new_hyps) == beam_size:
                    break
            assert len(new_hyps) == beam_size
            curr_hyps = new_hyps
            if self.lut.weight.data.is_cuda:
                inps = inps.cuda()
            embs = self.lut(inps).view(1, beam_size, -1)  # 1 x K x rnninsz

            cond_embs = torch.cat([embs, self.state_embs[rul_ss].expand(1, beam_size, state_emb_sz)], 2)
            hid, (hc, cc) = self.seg_rnn(cond_embs, (torch.cat(anc_hs, 1), torch.cat(anc_cs, 1)))
        # hypotheses of length L still need their end probs added
        # N.B. if the <eos> falls off the beam we could end up with situations
        # where we take an L-length phrase w/ a lower score than 1-word followed by eos.
        wrd_dist = self.get_next_word_dist(hid, rul_ss).cpu()  # K x nwords
        wrd_dist.log_()
        wrd_dist.add_(curr_scores.expand_as(wrd_dist))
        for k in range(beam_size):
            wlenscore = wrd_dist[k][self.eop_idx] / (self.L + 1) + len_lps[ss][self.L - 1]
            if wlenscore > best_hyp_score:
                best_hyp_score = wlenscore
                best_hyp = backtrace(curr_hyps[k])
                best_wscore, best_lscore = wrd_dist[k][self.eop_idx], len_lps[ss][self.L - 1]

        return best_hyp, best_wscore, best_lscore

    def gen_one(self, templt, h0, c0, src_sent_enc, len_lps, row2tblent, row2feats):
        """
        src - 1 x nfields x nfeatures
        h0 - rnn_size vector
        c0 - rnn_size vector
        src_sent_enc - 1 x src_seq_len x dim
        len_lps - K x L, log normalized
        returns a list of phrases
        """
        phrases = []
        tote_wscore, tote_lscore, tokes, segs = 0.0, 0.0, 0.0, 0.0
        # start_inp = self.lut.weight[start_idx].view(1, 1, -1)
        start_inp = self.start_emb
        exh0 = h0.view(1, 1, self.hid_size).expand(self.layers, 1, self.hid_size)
        exc0 = c0.view(1, 1, self.hid_size).expand(self.layers, 1, self.hid_size)
        nout_wrds = self.decoder.out_features
        i2w, w2i = corpus.dictionary.idx2word, corpus.dictionary.word2idx
        for stidx, k in enumerate(templt):
            phrs_idxs, wscore, lscore = self.temp_bs(k, start_inp, exh0, exc0,
                                                     len_lps, row2tblent, row2feats,
                                                     args.beamsz, final_state=(stidx == len(templt) - 1))
            phrs = []
            for ii in range(len(phrs_idxs)):
                if phrs_idxs[ii] < nout_wrds:
                    try:
                        phrs.append(i2w[phrs_idxs[ii]])
                    except:
                        phrs.append('<unk phrase>')
                else:
                    tblidx = phrs_idxs[ii] - nout_wrds
                    _, _, wordstr = row2tblent[tblidx]
                    if args.verbose:
                        phrs.append(wordstr + " (c)")
                    else:
                        phrs.append(wordstr)
            if phrs[-1] == "<eos>":
                break
            phrases.append(phrs)
            tote_wscore += wscore
            tote_lscore += lscore
            tokes += len(phrs_idxs) + 1  # add 1 for <eop> token
            segs += 1

        return phrases, tote_wscore, tote_lscore, tokes, segs

    def forward(self, inps, combotargs, constrs=None, idx=None, vocab_masks=None):
        '''
        HSMM forward.
        Args:
            inps: bsz x seq_len
            fmask: bsz x src_seq_len
            combotargs: bsz x L x seqlen
            constr: bsz list
            vocab_masks: 3 x gentypes python list
        Returns:

        '''
        bsz, L, seqlen = combotargs.size()
        if constrs:
            constrs = [constrs[int(_)] for _ in list(idx)]
            cidxs = make_bwd_constr_idxs(args.L, seqlen, constrs, args.seg_cut)
            if cidxs:
                cidxs = [tens.cuda() if tens is not None else None for tens in cidxs]
        else:
            cidxs = None

        if vocab_masks:
            vocab_masks = torch.Tensor(vocab_masks).cuda()
            vocab_masks.requires_grad = False

        inps = inps.transpose(0, 1)
        combotargs = combotargs.transpose(0, 1).contiguous().view(L, bsz * seqlen)
        # src_enc, src_sen_enc = self.encode(src)
        init_logps, trans_logps = self.trans_logprobs(bsz, seqlen)  # bsz x K, T-1 x bsz x KxK
        len_logprobs, _ = self.len_logprobs()
        fwd_obs_logps = self.obs_logprobs(inps, combotargs, vocab_masks)  # L x T x bsz x K
        # get T+1 x bsz x K beta quantities
        if args.original_bwd:
            beta, beta_star = infc._just_bwd(trans_logps, fwd_obs_logps, len_logprobs, constraints=cidxs)
        else:
            beta, beta_star = infc.just_bwd(trans_logps, fwd_obs_logps, len_logprobs, constraints=cidxs)
        log_marg = logsumexp1(beta_star[0] + init_logps).sum()  # bsz x 1 -> 1
        return log_marg

    def load_model(self, save_path):
        saved_stuff = torch.load(save_path)
        clean_state = {}
        saved_state = saved_stuff["state_dict"]
        for k, v in saved_state.items():
            nk = k[7:] if k.startswith('module.') else k
            clean_state[nk] = v
        self.load_state_dict(clean_state, strict=True)

def make_targs(x, L, ngen_types):
    """
    :param x: seqlen x bsz
    :param L:
    :param ngen_types:
    :return: L x bsz*seqlen tensor
    """
    seqlen, bsz = x.size()
    newlocs = torch.LongTensor(L, seqlen, bsz).fill_(ngen_types + 1)
    for i in range(L):
        newlocs[i][:seqlen - i].copy_(x[i:])
    # return newlocs.transpose(1, 2).contiguous().view(L, bsz * seqlen)
    return newlocs.permute(2, 0, 1).contiguous()  # bsz x L x seqlen



parser = argparse.ArgumentParser(description='')
parser.add_argument('-data', type=str, default='', help='path to data dir')
parser.add_argument('-epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('-bsz', type=int, default=16, help='batch size')
parser.add_argument('-seed', type=int, default=1111, help='random seed')
parser.add_argument('-cuda', action='store_true', help='use CUDA')
parser.add_argument('-log_interval', type=int, default=200,
                    help='minibatches to wait before logging training status')
parser.add_argument('-save', type=str, default='', help='path to save the final model')
parser.add_argument('-load', type=str, default='', help='path to saved model')
parser.add_argument('-vocab_path', type=str, default='question_data/vocab.data', help='the vocab file.')
parser.add_argument('-vocab_size', type=int, default=20000, help='the vocab size.')
parser.add_argument('-test', action='store_true', help='use test data')
parser.add_argument('-thresh', type=int, default=9, help='prune if occurs <= thresh')
parser.add_argument('-max_mbs_per_epoch', type=int, default=1e6, help='max minibatches per epoch')

parser.add_argument('-emb_size', type=int, default=300, help='size of word embeddings')
parser.add_argument('-hid_size', type=int, default=300, help='size of rnn hidden state')
parser.add_argument('-layers', type=int, default=1, help='num rnn layers')
parser.add_argument('-A_dim', type=int, default=64,
                    help='dim of factors if factoring transition matrix')
parser.add_argument('-cond_A_dim', type=int, default=32,
                    help='dim of factors if factoring transition matrix')
parser.add_argument('-smaller_cond_dim', type=int, default=64,
                    help='dim of thing we feed into linear to get transitions')
parser.add_argument('-yes_self_trans', action='store_true', help='')
parser.add_argument('-mlpinp', action='store_true', help='')
parser.add_argument('-max_pool', action='store_true', help='for word-fields')

parser.add_argument('-constr_tr_epochs', type=int, default=100, help='')
parser.add_argument('-no_ar_epochs', type=int, default=100, help='')
parser.add_argument('-unif_trans', action='store_true', help='unif prob for init transition.')

parser.add_argument('-word_ar', action='store_true', help='')
parser.add_argument('-ar_after_decay', action='store_true', help='autoregressive model')
parser.add_argument('-no_ar_for_vit', action='store_true', help='')
parser.add_argument('-fine_tune', action='store_true', help='only train ar rnn')
parser.add_argument('-seg_cut', action='store_true', help='whether cut segment necessarily.')
parser.add_argument('-no_constr', action='store_true', help='without constraints.')
parser.add_argument('-no_mask', action='store_true', help='no vocab mask')
parser.add_argument('-original_bwd', action='store_true', help='original bwd')


parser.add_argument('-dropout', type=float, default=0.3, help='dropout')
parser.add_argument('-emb_drop', action='store_true', help='dropout on embeddings')
parser.add_argument('-sep_attn', action='store_true', help='')
parser.add_argument('-max_seqlen', type=int, default=35, help='the max sequence length')

parser.add_argument('-K', type=int, default=10, help='number of states')
parser.add_argument('-Kmul', type=int, default=1, help='number of states multiplier')
parser.add_argument('-L', type=int, default=4, help='max segment length')
parser.add_argument('-unif_lenps', action='store_true', help='')
parser.add_argument('-one_rnn', action='store_true', help='')

parser.add_argument('-initrange', type=float, default=0.1, help='uniform init interval')
parser.add_argument('-lr', type=float, default=0.5, help='initial learning rate')
parser.add_argument('-lr_decay', type=float, default=0.5, help='learning rate decay')
parser.add_argument('-optim', type=str, default="sgd", help='optimization algorithm')
parser.add_argument('-onmt_decay', action='store_true', help='')
parser.add_argument('-clip', type=float, default=5, help='gradient clipping')
parser.add_argument('-interactive', action='store_true', help='')
parser.add_argument('-label_data', action='store_true', help='')
parser.add_argument('-split', type=str, default='train', help='the labeled data split')
parser.add_argument('-gen_from_fi', type=str, default='', help='')
parser.add_argument('-verbose', action='store_true', help='')
parser.add_argument('-prev_loss', type=float, default=None, help='')
parser.add_argument('-best_loss', type=float, default=None, help='')

parser.add_argument('-tagged_fi', type=str, default='', help='path to tagged fi')
parser.add_argument('-ntemplates', type=int, default=200, help='num templates for gen')
parser.add_argument('-beamsz', type=int, default=1, help='')
parser.add_argument('-gen_wts', type=str, default='1,9', help='')
parser.add_argument('-min_gen_tokes', type=int, default=5, help='')
parser.add_argument('-min_gen_states', type=int, default=3, help='')
parser.add_argument('-gen_on_valid', action='store_true', help='')
parser.add_argument('-align', action='store_true', help='')
parser.add_argument('-wid_workers', type=str, default='', help='')
# for analysis
parser.add_argument('-whole-res', type=str, default='whole_res.txt')
parser.add_argument('-temps', type=str, default='templates')

parser.add_argument('--gpu', type=str)
parser.add_argument('--dataDir', type=str)
parser.add_argument('--modelDir', type=str)
parser.add_argument('--logDir', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    print(vars(args))
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with -cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    if args.no_mask:
        vocab = datasets.Dictionary()
        vocab.load_from_file(args.vocab_path, args.vocab_size)
    else:
        vocab = datasets.MaskDictionary()
        vocab.load_from_file(ordinary_file='question_data/ordinary_vocab.data',
                             topic_file='question_data/topic_vocab.data',
                             interrogative_file='question_data/interrogative_vocab.data',
                             size=args.vocab_size)
    # Load data
    corpus = datasets.Corpus(args.data, args.bsz, vocab, add_bos=False, add_eos=False, quick=args.gen_from_fi)

    saved_args, saved_state, saved_epoch = None, None, None
    if len(args.load) > 0:
        print('load model from: ', args.load)
        saved_stuff = torch.load(args.load)
        if not args.best_loss:
            args.__dict__["best_loss"] = saved_stuff["best_valloss"]
            args.__dict__["prev_loss"] = saved_stuff["prev_valloss"]
        saved_args, saved_state, saved_epoch = saved_stuff["opt"], saved_stuff["state_dict"], saved_stuff["epoch"]
        for k, v in args.__dict__.items():
            if k not in saved_args.__dict__:
                saved_args.__dict__[k] = v
        net = HSMM(len(corpus.dictionary), corpus.ngen_types, saved_args)

        # remove the 'module' prefix
        clean_state = {}
        for k, v in saved_state.items():
            nk = k[7:] if k.startswith('module.') else k
            clean_state[nk] = v
        net.load_state_dict(clean_state, strict=True)
        args.pad_idx = corpus.dictionary.word2idx["<pad>"]
        if args.fine_tune:
            for name, param in net.named_parameters():
                if name in saved_state:
                    param.requires_grad = False

    else:
        args.pad_idx = corpus.dictionary.word2idx["<pad>"]
        net = HSMM(len(corpus.dictionary), corpus.ngen_types, args)

    if torch.cuda.device_count() > 1 and not args.label_data and not args.gen_from_fi:
        net = torch.nn.DataParallel(net).cuda()
        print('cuda ids', net.device_ids)

    elif args.cuda:
        net = net.cuda()

    if args.optim == "adagrad":
        optalg = optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
        for group in optalg.param_groups:
            for p in group['params']:
                optalg.state[p]['sum'].fill_(0.1)
    elif args.optim == "rmsprop":
        optalg = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    elif args.optim == "adam":
        optalg = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    else:
        optalg = None


    def save_model(args, net, save_path):
        state = {"opt": args, "state_dict": net.state_dict(),
                 "lr": args.lr, "dict": corpus.dictionary,
                 "epoch": epoch, "best_valloss": best_valloss, "prev_valloss": prev_valloss}
        save_path = save_path
        torch.save(state, save_path)


    def train(epoch):
        # Turn on training mode which enables dropout.
        net.train()
        neglogev = 0.0  # negative log evidence
        nsents = 0
        latest_neglogev = 0.0

        trainperm = torch.randperm(len(corpus.train))
        nmini_batches = len(corpus.train)
        for batch_idx in range(nmini_batches):
            # save model
            if batch_idx % 5000 == 0:
                save_model(args, net, args.save + '.b_{}'.format(batch_idx))

            net.zero_grad()

            x, constrs, src, inps = corpus.train[trainperm[batch_idx]]

            inps = inps.t()
            if args.no_constr:
                constrs = None

            seqlen, bsz = x.size()
            if seqlen < args.L or seqlen > args.max_seqlen or (args.cuda and bsz < torch.cuda.device_count()):
                continue

            combotargs = make_targs(x, args.L, corpus.ngen_types)  # bsz x L x seqlen
            # get bsz x src_len, bsz x src_len masks
            # fmask, amask = make_src_masks(src, args.pad_idx)

            if args.cuda:
                combotargs = combotargs.cuda()
                src = src.cuda()
                inps = inps.cuda()
                # fmask, amask = fmask.cuda(), amask.cuda()

            vocab_masks = None
            log_marg = net(inps, combotargs, constrs, torch.arange(bsz), vocab_masks)

            # calculate loss
            log_marg = log_marg.sum()

            if log_marg.item() < -1000 * bsz:  # remove outliers
                print('outlier batch')
                continue

            lossvar = -log_marg / bsz
            # print(lossvar)
            lossvar.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), args.clip)
            if optalg is not None:
                optalg.step()
            else:
                for p in net.parameters():
                    if p.grad is not None:
                        p.data.add_(-args.lr, p.grad.data)

            neglogev -= log_marg.item()
            nsents += bsz

            if (batch_idx + 1) % args.log_interval == 0:
                print("batch %d/%d | train neglogev %g " % (batch_idx + 1,
                                                            nmini_batches,
                                                            neglogev / nsents))
                latest_neglogev = neglogev / nsents
                neglogev, nsents = 0, 0

        epoch_neglogev = neglogev / nsents if nsents else latest_neglogev
        print("epoch %d | train neglogev %g " % (epoch, epoch_neglogev))
        return epoch_neglogev


    def test_batch(x, src, inps):
        cidxs = None
        # seqlen, bsz = x.size()
        with torch.no_grad():
            combotargs = make_targs(x, args.L, corpus.ngen_types)
            # get bsz x src_len, bsz x src_len masks
            # fmask, amask = make_src_masks(src, args.pad_idx)

            if args.cuda:
                combotargs = combotargs.cuda()
                if cidxs is not None:
                    cidxs = [tens.cuda() if tens is not None else None for tens in cidxs]
                src = src.cuda()
                inps = inps.cuda()
                # fmask, amask = fmask.cuda(), amask.cuda()

            # vocab_masks = [vocab.interrogative_mask, vocab.topic_mask, vocab.ordinary_mask]
            log_marg = net(inps, combotargs, None, None, None)
            log_marg = log_marg.sum()
            return float(log_marg.data)


    def test(epoch):
        net.eval()
        neglogev = 0.0
        nsents = 0

        for i in range(len(corpus.valid)):
            x, _, src, inps = corpus.valid[i]
            inps = inps.t()
            cidxs = None

            seqlen, bsz = x.size()
            if seqlen < args.L or seqlen > args.max_seqlen:
                continue

            lma = test_batch(x, src, inps)
            neglogev -= lma
            nsents += bsz
        print("epoch %d | valid ev %g" % (epoch, neglogev / nsents))
        return neglogev / nsents


    def label_data(split='train'):
        id2w = corpus.dictionary.idx2word
        w2id = corpus.dictionary.word2idx

        with torch.no_grad():
            dataset = corpus.train if split == 'train' else corpus.valid
            print(f"{split} size:", len(dataset))
            for i in range(len(dataset)):
                x, _, src, inps = dataset[i]

                seqlen, bsz = x.size()
                args.__dict__["max_seqlen"] = 10
                if seqlen <= saved_args.L or seqlen > args.max_seqlen:
                    continue

                combotargs = make_targs(x, args.L, corpus.ngen_types)

                if args.cuda:
                    combotargs = combotargs.cuda()
                    src = src.cuda()
                    inps = inps.cuda()

                init_logps, trans_logps = net.trans_logprobs(bsz, seqlen)  # bsz x K, T-1 x bsz x KxK
                len_logprobs, _ = net.len_logprobs()
                fwd_obs_logps = net.obs_logprobs(inps, combotargs, None)  # L x T x bsz x K
                bwd_obs_logprobs = infc.bwd_from_fwd_obs_logprobs(fwd_obs_logps.data)
                seqs = infc.viterbi(init_logps.data, trans_logps.data, bwd_obs_logprobs,
                                    [t.data for t in len_logprobs])
                for b in range(bsz):
                    src_words = [id2w[w] for w in src[b] if w != w2id['<pad>']]
                    words = [id2w[w] for w in x[:, b]]
                    print(' '.join(src_words), end=' ||| ')
                    for (start, end, label) in seqs[b]:
                        # print(start, end)
                        print("%s|%d " % (" ".join(words[start:end]), label), end="")
                    print()


    def gen_from_srctbl(src_sent, top_temps, coeffs, src_line=None):
        '''

        Args:
            src_sent: src_seq index list
            top_temps: top templates
            coeffs:
            src_line:

        Returns:

        '''
        # net.ar = saved_args.ar_after_decay
        net.ar = False
        # print "btw2", net.ar
        i2w, w2i = corpus.dictionary.idx2word, corpus.dictionary.word2idx
        best_score, best_phrases, best_templt = -float("inf"), None, None
        best_len = 0
        best_tscore, best_gscore = None, None

        src_seq = torch.LongTensor(src_sent).unsqueeze(dim=0)  # 1 x src_seq_len
        if args.cuda:
            src_seq = src_seq.cuda()

        src_enc, src_sen_enc = None, None
        init_logps, trans_logps = net.trans_logprobs(1, 2)
        _, len_scores = net.len_logprobs()
        len_lps = net.lsm(len_scores).data.cpu()
        init_logps, trans_logps = init_logps.data.cpu(), trans_logps.data[0].cpu()
        # inits = net.h0_lin
        inits = net.h0_lin.unsqueeze(0)
        h0, c0 = F.tanh(inits[:, :inits.size(1) // 2]), inits[:, inits.size(1) // 2:]
        # select template from trans_log_probs
        top_temps = []
        bb = []
        # bb = [13,2,7,30,27,43,14,38]
        dd = []
        for b in bb:
            dd += [50 * _ + b for _ in range(4)]

        t_lps = trans_logps[0]  # K x K
        for d in dd:
            t_lps[:, d] = float("-inf")
            init_logps[:, d] = float("-inf")

        _, states = init_logps[0].topk(50)
        states = states[-10:]
        for state in states:
            templt = []
            templt.append(state)
            while len(templt) < 4:
                _, state = t_lps[state].topk(10)
                state = state[-1]
                templt.append(state)
            top_temps.append(templt)

            # #find templt in top templates
            # # print('The template:', templt)
            # for i in range(3,5):
            #     top_temps.append(templt[:i])
            # # print('top templts:', top_temps)
        constr_sat = False
        # search over all templates
        res = []
        for templt in top_temps:
            # print "templt is", templt
            # get templt transition prob
            tscores = [init_logps[0][templt[0]]]
            [tscores.append(trans_logps[0][templt[tt - 1]][templt[tt]])
             for tt in range(1, len(templt))]

            if net.ar:
                phrases, wscore, tokes = net.gen_one_ar(templt, h0[0], c0[0], src_sen_enc,
                                                        len_lps, None, None)
                rul_tokes = tokes
            else:
                phrases, wscore, lscore, tokes, segs = net.gen_one(templt, h0[0], c0[0],
                                                                   src_sen_enc, len_lps, None, None)
                rul_tokes = tokes - segs  # subtract imaginary toke for each <eop>
                wscore /= tokes
            segs = len(templt)
            if (rul_tokes < args.min_gen_tokes or segs < args.min_gen_states) and constr_sat:
                continue
            if rul_tokes >= args.min_gen_tokes and segs >= args.min_gen_states:
                constr_sat = True  # satisfied our constraint
            tscore = sum(tscores[:int(segs)]) / segs
            if not net.unif_lenps:
                tscore += lscore / segs

            # for output.
            gq = ' '.join([' '.join(_) for _ in phrases])
            tmpltd = ' '.join(["%s|%d" % (' '.join(phrs), templt[kk]) for kk, phrs in enumerate(phrases)])
            res.append(gq + ' ||| ' + tmpltd + ' ||| ' + f'{tscore:.4f}_{wscore:.4f}')

            gscore = wscore
            # ascore=gscore
            ascore = coeffs[0] * tscore + coeffs[1] * gscore
            if (constr_sat and ascore > best_score) or (not constr_sat and rul_tokes > best_len) or (
                    not constr_sat and rul_tokes == best_len and ascore > best_score):
                # take if improves score or not long enough yet and this is longer...
                # if ascore > best_score: #or (not constr_sat and rul_tokes > best_len):
                best_score, best_tscore, best_gscore = ascore, tscore, gscore
                best_phrases, best_templt = phrases, templt
                best_len = rul_tokes
            # str_phrases = [" ".join(phrs) for phrs in phrases]
            # tmpltd = ["%s|%d" % (phrs, templt[k]) for k, phrs in enumerate(str_phrases)]
            # statstr = "a=%.2f t=%.2f g=%.2f" % (ascore, tscore, gscore)
            # print "%s|||%s" % (" ".join(str_phrases), " ".join(tmpltd)), statstr
            # assert False
        # assert False

        try:
            str_phrases = [" ".join(phrs) for phrs in best_phrases]
        except TypeError:
            # sometimes it puts an actual number in
            str_phrases = [" ".join([str(n) if type(n) is int else n for n in phrs]) for phrs in best_phrases]
        tmpltd = ["%s|%d" % (phrs, best_templt[kk]) for kk, phrs in enumerate(str_phrases)]
        if args.verbose:
            print(src_line)
            # print src_tbl

        # print("%s|||%s" % (" ".join(str_phrases), " ".join(tmpltd)))
        tokens = src_line.strip().split()
        l = []
        for t in tokens:
            if t not in w2i:
                l.append('_' + t + '_')
            else:
                l.append(t)
        src_line = ' '.join(l)
        print("%s|||%s|||%s" % (src_line, " ".join(str_phrases), " ".join(tmpltd)))
        if args.verbose:
            statstr = "a=%.2f t=%.2f g=%.2f" % (best_score, best_tscore, best_gscore)
            print(statstr)
            print()
        # assert False
        return res


    def gen_from_src():
        from template_extraction import extract_from_tagged_data, align_cntr
        top_temps, temps2sents, state2phrases = extract_from_tagged_data(args.tagged_fi, args.ntemplates)

        with open(args.temps + '_tmp2sent.txt', 'w', encoding='utf8')as p:
            import json
            tmp = {}
            for k, v in temps2sents.items():
                v = [' '.join(_[0]) for _ in v]
                tmp[str(k)] = v
            json.dump(tmp, p, indent=2, ensure_ascii=False)
        with open(args.temps + '_sta2phr.txt', 'w', encoding='utf8')as p:
            import json
            tmp = {}
            for k, v in state2phrases.items():
                tmp[k] = (v[0], [_.item() for _ in v[1]])
            json.dump(tmp, p, indent=2, ensure_ascii=False)
        print('templates dumped.')

        args.gen_from_fi = '/mnt/tobey/STC/dataset/QGdata/weibo_pair_dev_Q.post'
        with open(args.gen_from_fi) as f:
            src_lines = [_.strip() for _ in f.readlines()][:200]
            whole_res = {}

        net.eval()
        coeffs = [float(flt.strip()) for flt in args.gen_wts.split(',')]
        for ll, src_line in enumerate(src_lines):
            src_sent = [corpus.dictionary.add_word(_) for _ in src_line.strip().split()]
            res = gen_from_srctbl(src_sent, top_temps, coeffs, src_line=src_line)
            whole_res[src_line] = res

        with open(args.whole_res, 'w', encoding='utf8')as p:
            import json
            json.dump(whole_res, p, indent=2, ensure_ascii=False)


    if args.interactive:
        pass
    elif args.label_data:
        net.eval()
        print('begin to label train set.')
        label_data(args.split)
    elif len(args.gen_from_fi) > 0:
        gen_from_src()
    elif args.epochs == 0:
        net.eval()
        test(0)
    else:
        prev_valloss, best_valloss = float("inf"), float("inf")
        decayed = False
        if args.prev_loss is not None:
            prev_valloss = args.prev_loss
            if args.best_loss is None:
                best_valloss = prev_valloss
            else:
                decayed = True
                best_valloss = args.best_loss
            print("starting with", prev_valloss, best_valloss)

        epoch = 1 if not saved_epoch else saved_epoch + 1
        while epoch < args.epochs + 1:
            train(epoch)
            net.eval()
            valloss = test(epoch)

            if len(args.save) > 0:
                state = {"opt": args, "state_dict": net.state_dict(),
                         "lr": args.lr, "dict": corpus.dictionary,
                         "epoch": epoch, "best_valloss": best_valloss, "prev_valloss": prev_valloss}
                save_path = args.save + '.e{}'.format(epoch)
                torch.save(state, save_path)
                print("saving to", save_path)

                if valloss < best_valloss:
                    best_valloss = valloss
                    save_path = args.save + '.best'
                    torch.save(state, save_path)
                    print("Updated best model.")
            if valloss >= prev_valloss or decayed:
                decayed = True
                args.lr *= args.lr_decay
                print("decaying lr to:", args.lr)
                if args.lr < 1e-5 and False:
                    break
            prev_valloss = valloss
            epoch += 1


'''
python chsmm_without_src.py -data question_data/zhihu_data/ -bsz 16 -cuda -emb_size 300 -hid_size 300 -L 4 -K 50 -max_seqlen 35 -vocab_path question_data/vocab.data -seg_cut -unif_lenps -no_mask -save models/hsmm-300-50-4.pt

python chsmm_without_src.py -label_data -split train -data question_data/zhihu_data/ -bsz 16 -cuda -emb_size 300 -hid_size 300 -L 4 -K 50 -max_seqlen 35 -vocab_path question_data/vocab.data -seg_cut -unif_lenps -no_mask -load models/hsmm-300-50-4.pt.e10 | tee models/hsmm-300-50-4-far.pt.e10.train.segs

python chsmm_without_src.py -label_data -split valid -data question_data/zhihu_data/ -bsz 16 -cuda -emb_size 300 -hid_size 300 -L 4 -K 50 -max_seqlen 35 -vocab_path question_data/vocab.data -seg_cut -unif_lenps -no_mask -load models/hsmm-300-50-4.pt.e10 | tee models/hsmm-300-50-4-far.pt.e10.valid.segs

'''