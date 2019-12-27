"""
utils and whatnot
"""
import math
from collections import defaultdict, Counter
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random

def logsumexp0(X):
    """
    X - L x B x K
    returns:
        B x K
    """
    if X.dim() == 2:
        X = X.unsqueeze(2)
    axis = 0
    X2d = X.view(X.size(0), -1)
    maxes, _ = torch.max(X2d, axis, True)
    lse = maxes + torch.log(torch.sum(torch.exp(X2d - maxes.expand_as(X2d)), axis, True))
    lse = lse.view(X.size(1), -1)
    return lse


def logsumexp2(X):
    """
    X - L x B x K
    returns:
        L x B
    """
    if X.dim() == 2:
        X = X.unsqueeze(0)
    X2d = X.view(-1, X.size(2))
    maxes, _ = torch.max(X2d, 1, True)
    lse = maxes + torch.log(torch.sum(torch.exp(X2d - maxes.expand_as(X2d)), 1, True))
    lse = lse.view(X.size(0), -1)
    return lse


def logsumexp1(X):
    """
    X - B x K
    returns:
        B x 1
    """
    maxes, _ = torch.max(X, 1, True)
    lse = maxes + torch.log(torch.sum(torch.exp(X - maxes.expand_as(X)), 1, True))
    return lse


def vlogsumexp(v):
    """
    for vectors
    """
    maxv = v.max()
    return maxv + math.log(torch.sum(torch.exp(v - maxv)))


def make_fwd_constr_idxs(L, T, constrs):
    """
    for use w/ fwd alg.
    constrs are 0-indexed
    """
    cidxs = [set() for t in range(T)]
    bsz = len(constrs)
    for b in range(bsz):
        for tup in constrs[b]:
            if len(tup) == 2:
                start, end = tup
            else:
                start, end = tup[0], tup[1]
            clen = end - start
            # for last thing in segment only allow segment length
            end_steps_back = min(L, end)
            cidxs[end - 1].update([(end_steps_back - l - 1) * bsz + b
                                   for l in range(end_steps_back) if l + 1 != clen])
            # now disallow everything for everything else in the segment
            for i in range(start, end - 1):
                steps_back = min(L, i + 1)
                cidxs[i].update([(steps_back - l - 1) * bsz + b for l in range(steps_back)])
            # now disallow things w/in L of the end
            for i in range(end, min(T, end + L - 1)):
                steps_back = min(L, i + 1)
                cidxs[i].update([(steps_back - l + end - 1) * bsz + b for l in range(i + 1, end + steps_back)])
    oi_cidxs = [None]  # make 1-indexed
    oi_cidxs.extend([torch.LongTensor(list(idxs)) if len(idxs) > 0 else None for idxs in cidxs])
    return oi_cidxs


def make_bwd_constr_idxs(L, T, constrs, seg_cut=False):
    """
    for use w/ bwd alg.
    constrs are a bsz-length list of lists of (start, end, label) 0-indexed tups
    """
    cidxs = [set() for t in range(T)]
    bsz = len(constrs)
    for b in range(bsz):
        for tup in constrs[b]:
            if len(tup) == 2:
                start, end = tup
            else:
                start, end = tup[0], tup[1]
            clen = end - start
            steps_fwd = min(L, T - start)
            # # for first thing only allow segment length
            # cidxs[start].update([l*bsz + b for l in range(steps_fwd) if l+1 != clen])

            # tobey: change to allow be equal or larger than segment length.
            if seg_cut:
                cidxs[start].update([l * bsz + b for l in range(steps_fwd) if l + 1 != clen])
            else:
                cidxs[start].update([l * bsz + b for l in range(steps_fwd) if l + 1 < clen])

            # now disallow everything for everything else in the segment
            for i in range(start + 1, end):
                steps_fwd = min(L, T - i)
                cidxs[i].update([l * bsz + b for l in range(steps_fwd)])

            # now disallow things w/in L of the start
            for i in range(max(start - L + 1, 0), start):
                steps_fwd = min(L, T - i)
                # cidxs[i].update([l*bsz + b for l in range(steps_fwd) if i+l >= start])
                # tobey:
                if seg_cut:
                    cidxs[i].update([l * bsz + b for l in range(steps_fwd) if i + l >= start])
                else:
                    cidxs[i].update([l * bsz + b for l in range(steps_fwd) if i + l > start and i + l < end])

    oi_cidxs = [None]
    oi_cidxs.extend([torch.LongTensor(list(idxs)) if len(idxs) > 0 else None for idxs in cidxs])

    return oi_cidxs


def backtrace(node):
    """
    assumes a node is (word, node) and that every history starts with (None, None)
    """
    hyp = [node[0]]
    while node[1] is not None:
        node = node[1]
        hyp.append(node[0])
    return hyp[-2::-1]  # returns all but last element, reversed


def backtrace3(node):
    """
    assumes a node is (word, seg-label, node) etc
    """
    hyp = [(node[0], node[1])]
    while node[2] is not None:
        node = node[2]
        hyp.append((node[0], node[1]))
    return hyp[-2::-1]


def beam_search2(net, corpus, ss, start_inp, exh0, exc0, srcfieldenc,
                 len_lps, row2tblent, row2feats, K, final_state=False):
    """
    ss - discrete state index
    exh0 - layers x 1 x rnn_size
    exc0 - layers x 1 x rnn_size
    start_inp - 1 x 1 x emb_size
    len_lps - K x L, log normalized
    """
    rul_ss = ss % net.K
    i2w = corpus.dictionary.idx2word
    w2i = corpus.dictionary.word2idx
    genset = corpus.genset
    unk_idx, eos_idx, pad_idx = w2i["<unk>"], w2i["<eos>"], w2i["<pad>"]
    state_emb_sz = net.state_embs.size(3) if net.one_rnn else 0
    if net.one_rnn:
        cond_start_inp = torch.cat([start_inp, net.state_embs[rul_ss]], 2)  # 1 x 1 x cat_size
        hid, (hc, cc) = net.seg_rnns[0](cond_start_inp, (exh0, exc0))
    else:
        hid, (hc, cc) = net.seg_rnns[rul_ss](start_inp, (exh0, exc0))
    curr_hyps = [(None, None)]
    best_wscore, best_lscore = None, None  # so we can truly average over words etc later
    best_hyp, best_hyp_score = None, -float("inf")
    curr_scores = torch.zeros(K, 1)
    # N.B. we assume we have a single feature row for each timestep rather than avg
    # over them as at training time. probably better, but could conceivably average like
    # at training time.
    inps = Variable(torch.LongTensor(K, 4), volatile=True)
    for ell in range(net.L):
        wrd_dist = net.get_next_word_dist(hid, rul_ss, srcfieldenc).cpu()  # K x nwords
        # disallow unks
        wrd_dist[:, unk_idx].zero_()
        if not final_state:
            wrd_dist[:, eos_idx].zero_()
        # if not ss == 25 or not ell == 3:
        net.collapse_word_probs(row2tblent, wrd_dist)
        wrd_dist.log_()
        if ell > 0:  # add previous scores
            wrd_dist.add_(curr_scores.expand_as(wrd_dist))
        maxprobs, top2k = torch.topk(wrd_dist.view(-1), 2 * K)
        cols = wrd_dist.size(1)
        # we'll break as soon as <eos> is at the top of the beam.
        # this ignores <eop> but whatever
        if top2k[0] == eos_idx:
            final_hyp = backtrace(curr_hyps[0])
            final_hyp.append(eos_idx)
            return final_hyp, maxprobs[0], len_lps[ss][ell]

        new_hyps, anc_hs, anc_cs = [], [], []
        # inps.data.fill_(pad_idx)
        inps.data[:, 1].fill_(w2i["<ncf1>"])
        inps.data[:, 2].fill_(w2i["<ncf2>"])
        inps.data[:, 3].fill_(w2i["<ncf3>"])
        for k in range(2 * K):
            anc, wrd = top2k[k] / cols, top2k[k] % cols
            # check if any of the maxes are eop
            if wrd == net.eop_idx and ell > 0:
                # add len score (and avg over num words incl eop i guess)
                wlenscore = maxprobs[k] / (ell + 1) + len_lps[ss][ell - 1]
                if wlenscore > best_hyp_score:
                    best_hyp_score = wlenscore
                    best_hyp = backtrace(curr_hyps[anc])
                    best_wscore, best_lscore = maxprobs[k], len_lps[ss][ell - 1]
            else:
                curr_scores[len(new_hyps)][0] = maxprobs[k]
                if wrd >= net.decoder.out_features:  # a copy
                    tblidx = wrd - net.decoder.out_features
                    inps.data[len(new_hyps)].copy_(row2feats[tblidx])
                else:
                    inps.data[len(new_hyps)][0] = wrd if i2w[wrd] in genset else unk_idx
                new_hyps.append((wrd, curr_hyps[anc]))
                anc_hs.append(hc.narrow(1, anc, 1))  # layers x 1 x rnn_size
                anc_cs.append(cc.narrow(1, anc, 1))  # layers x 1 x rnn_size
            if len(new_hyps) == K:
                break
        assert len(new_hyps) == K
        curr_hyps = new_hyps
        if net.lut.weight.data.is_cuda:
            inps = inps.cuda()
        embs = net.lut(inps).view(1, K, -1)  # 1 x K x nfeats*emb_size
        if net.mlpinp:
            embs = net.inpmlp(embs)  # 1 x K x rnninsz
        if net.one_rnn:
            cond_embs = torch.cat([embs, net.state_embs[rul_ss].expand(1, K, state_emb_sz)], 2)
            hid, (hc, cc) = net.seg_rnns[0](cond_embs, (torch.cat(anc_hs, 1), torch.cat(anc_cs, 1)))
        else:
            hid, (hc, cc) = net.seg_rnns[rul_ss](embs, (torch.cat(anc_hs, 1), torch.cat(anc_cs, 1)))
    # hypotheses of length L still need their end probs added
    # N.B. if the <eos> falls off the beam we could end up with situations
    # where we take an L-length phrase w/ a lower score than 1-word followed by eos.
    wrd_dist = net.get_next_word_dist(hid, rul_ss, srcfieldenc).cpu()  # K x nwords
    # wrd_dist = net.get_next_word_dist(hid, ss, srcfieldenc).cpu() # K x nwords
    wrd_dist.log_()
    wrd_dist.add_(curr_scores.expand_as(wrd_dist))
    for k in range(K):
        wlenscore = wrd_dist[k][net.eop_idx] / (net.L + 1) + len_lps[ss][net.L - 1]
        if wlenscore > best_hyp_score:
            best_hyp_score = wlenscore
            best_hyp = backtrace(curr_hyps[k])
            best_wscore, best_lscore = wrd_dist[k][net.eop_idx], len_lps[ss][net.L - 1]
    # if ss == 80:
    #    print "going with", best_hyp
    return best_hyp, best_wscore, best_lscore


def batchwise_sample(set, generator, batch_samples, templates=None, max_rep_len=15, is_train=True, pad_idx=1):
    # perm = torch.randperm(len(set))
    # samples_ids = perm[:batch_samples]
    perm = list(range(len(set)))
    random.shuffle(perm)
    samples_ids = perm[:batch_samples]

    samples = []
    for i in samples_ids:
        query, pos_rep, template = set[perm[i]]
        query, pos_rep, template = query.cuda(), pos_rep.cuda(), template.cuda()
        bsz, rep_len = pos_rep.size()
        tpl_len = max(max_rep_len, rep_len)
        # #remove the <eos>
        # pos_rep[:,-1]=pad_idx
        pos_rep = F.pad(pos_rep,(0,tpl_len-rep_len),value=pad_idx)
        if is_train:
            tpls = torch.stack([generator.template_sample(templates, tpl_len) for _ in range(bsz)], 0).cuda()
        elif rep_len == tpl_len:
            tpls = template
        else:
            tpls = torch.cat([template, template.narrow(1,rep_len-1,1).expand(bsz, tpl_len-rep_len)],1)
        neg_rep = generator.sample_one(query, None, tpls)
        # neg_rep = generator.beam_sample(query, tpls)
        pos_target = torch.ones(bsz).cuda()
        neg_target = torch.zeros(bsz).cuda()
        query = torch.cat([query, query], 0)  # bsz*2 x query_len
        response = torch.cat([pos_rep, neg_rep], 0)  # bsz*2 x response_len
        target = torch.cat([pos_target, neg_target], 0)
        samples.append((query, response, target))
    return samples

# def
