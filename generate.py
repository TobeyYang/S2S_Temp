import torch, os
from tqdm import tqdm
from utils import backtrace
import argparse
import generator
from chsmm_without_src import HSMM
import numpy as np
import copy


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, source_words, log_probs, state, attn_dists):
        """Hypothesis constructor.

    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
      attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
      p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
      coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
    """
        self.tokens = tokens
        self.source_words = source_words
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists

    def extend(self, token, word, log_prob, state, attn_dist):
        """Return a NEW hypothesis, extended with the information from the latest step of beam search.

    Args:
      token: Integer. Latest token produced by beam search.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
      attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
    Returns:
      New Hypothesis for next step.
    """
        return Hypothesis(
            tokens=self.tokens + [token],
            source_words=self.source_words + [word],
            log_probs=self.log_probs + [log_prob],
            state=state,
            attn_dists=self.attn_dists + [attn_dist]
        )

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)  # every hyp pad [START] with 0.0 prob at first.

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        # return self.log_prob / max(len(self.tokens) - 1, 1)
        return self.log_prob / (np.power(5 + len(self.tokens), 0.6)/np.power(5+1, 0.6))
        # return self.log_prob


def beam_search(net, vocab, src, beam_size, template):
    i2w, w2i = vocab.idx2word, vocab.word2idx
    unk_idx, eos_idx, pad_idx = w2i["<unk>"], w2i["<eos>"], w2i["<pad>"]
    ll_idx, oov_idx = w2i.get("__",unk_idx), w2i.get('oov',unk_idx)

    src_enc, src_sent_enc = net.encode(src)
    src_mask = net.make_src_masks(src)
    inits = net.hsmm_net.h0_lin.unsqueeze(0)  # the fist rnn inits
    h1, c1 = torch.tanh(inits[:, :net.hsmm_net.hid_size]), inits[:, net.hsmm_net.hid_size:]
    h1 = h1.expand(1, 1, net.hsmm_net.hid_size).contiguous()
    c1 = c1.expand(1, 1, net.hsmm_net.hid_size).contiguous()

    h2, c2 = torch.tanh(src_enc), src_enc  # bsz x hid_size
    h2, c2 = h2.unsqueeze(0).contiguous(), c2.unsqueeze(0).contiguous()  # 1 x bsz x hid_size

    hyps = [Hypothesis(tokens=[], source_words=[], log_probs=[], state=(h1, c1, h2, c2), attn_dists=[], )]

    # inps = torch.LongTensor(beam_size)
    # inp_emb = net.hsmm_net.start_emb
    res = []
    min_len = 7
    for i in range(len(template)):
        if i == 0:
            inp_emb = net.hsmm_net.start_emb  # 1 x 1 x emb_size
        else:
            # inps = torch.LongTensor([h.latest_token for h in hyps]).cuda()
            inps = torch.stack([h.latest_token for h in hyps], 0)
            inp_emb = net.hsmm_net.lut(inps).unsqueeze(0)  # 1 x beam_size x emb_size
            h1 = torch.stack([_.state[0] for _ in hyps], 1)  # 1 x beam_size x hid_size
            c1 = torch.stack([_.state[1] for _ in hyps], 1)
            h2 = torch.stack([_.state[2] for _ in hyps], 1)
            c2 = torch.stack([_.state[3] for _ in hyps], 1)
            bsz, src_len, src_emb_size = src_sent_enc.size()
            if bsz < beam_size:
                src_sent_enc = src_sent_enc.expand(beam_size, src_len, src_emb_size)
                src_mask = src_mask.expand(beam_size, src_len)

        _, s, _ = inp_emb.size()  # s==1 or beam_size
        cur_ss = template[i]
        ss = torch.stack([cur_ss] * s)
        wrd_dist, h1, c1, h2, c2 = net.get_next_word_dist(inp_emb, ss, h1, c1, h2, c2, src_sent_enc, src_mask)
        wrd_dist = torch.softmax(wrd_dist[0], 1)  # bsz x V

        #
        wrd_dist[:, unk_idx].zero_()  # disallow unks
        wrd_dist[:, ll_idx].zero_()
        wrd_dist[:, oov_idx].zero_()

        # wrd_dist[:, net.hsmm_net.eop_idx].zero_()  # disallow the eop idx
        if cur_ss != template[-1] or i <= min_len:  # final state
            wrd_dist[:, eos_idx].zero_()

        all_hyps = []
        num_orig_hyps = 1 if i == 0 else len(hyps)
        for i in range(num_orig_hyps):
            orig_hyp, dist = hyps[i], wrd_dist[i]
            probs, topk_ids = torch.topk(dist, beam_size)
            for p, id in zip(probs, topk_ids):
                new_hyps = orig_hyp.extend(id, i2w[id], torch.log(p), (h1[:, i], c1[:, i], h2[:, i], c2[:, i]), None)
                all_hyps.append(new_hyps)

        hyps = []
        sorted_hyps = sort_hyps(all_hyps)
        for h in sorted_hyps:
            if h.latest_token == eos_idx:
                res.append(h)
            else:
                hyps.append(h)
            if len(hyps) == beam_size:
                break

        while len(hyps) < beam_size:
            hyps.append(copy.copy(hyps[-1]))

    if len(res) == 0:
        res = hyps
    res = sort_hyps(res)

    return res[0].tokens, res[0].avg_log_prob


def sort_hyps(hyps):
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)


def _beam_search(net, vocab, src, beam_size, template):
    '''
    beam search
    Args:
        net:
        vocab:
        ss:
        src: 1 x src_len
        beam_size:
        template: list of states. length 15
        final_state:

    Returns:

    '''
    # net = net.cpu()
    min_len = 5
    i2w, w2i = vocab.idx2word, vocab.word2idx
    unk_idx, eos_idx, pad_idx = w2i["<unk>"], w2i["<eos>"], w2i["<pad>"]

    # # split the inps into segs
    # begin = [sum(seg_lens[:i]) for i, _ in enumerate(seg_lens)]
    # end = [b + l for b, l in zip(begin, seg_lens)]

    src_enc, src_sent_enc = net.encode(src)
    src_mask = net.make_src_masks(src)

    curr_hyps = [(None, None)]
    curr_scores = torch.zeros(beam_size, 1)
    best_wscore, best_lscore = None, None  # so we can truly average over words etc later
    best_hyp, best_hyp_score = None, -float("inf")
    res = []

    inits = net.hsmm_net.h0_lin.unsqueeze(0)  # the fist rnn inits
    h1, c1 = torch.tanh(inits[:, :net.hsmm_net.hid_size]), inits[:, net.hsmm_net.hid_size:]
    h1 = h1.expand(1, 1, net.hsmm_net.hid_size).contiguous()
    c1 = c1.expand(1, 1, net.hsmm_net.hid_size).contiguous()

    h2, c2 = torch.tanh(src_enc), src_enc  # bsz x hid_size
    h2, c2 = h2.unsqueeze(0).contiguous(), c2.unsqueeze(0).contiguous()  # 1 x bsz x hid_size

    inps = torch.LongTensor(beam_size)
    inp_emb = net.hsmm_net.start_emb
    for i in range(len(template)):
        _, s, _ = inp_emb.size()  # s==1 or beamsize
        cur_ss = template[i]
        ss = torch.LongTensor([cur_ss] * s)
        wrd_dist, h1, c1, h2, c2 = net.get_next_word_dist(inp_emb, ss, h1, c1, h2, c2, src_sent_enc, src_mask)
        wrd_dist = torch.softmax(wrd_dist[0], 1)  # bsz x V
        wrd_dist[:, unk_idx].zero_()  # disallow unks
        # wrd_dist[:, net.hsmm_net.eop_idx].zero_()  # disallow the eop idx
        if cur_ss != template[-1] or i <= min_len:  # final state
            wrd_dist[:, eos_idx].zero_()

        # shield words.
        # if i == 0:
        #     sh_wrds=['我', '你', '谁', '这', '是', '那','什么', '难道']
        #     for _ in sh_wrds:
        #         wrd_dist[:, w2i[_]].zero_()

        wrd_dist.log_()
        if i > 0:  # add previous scores
            wrd_dist.add_(curr_scores.expand_as(wrd_dist))
        maxprobs, top2k = torch.topk(wrd_dist.view(-1), 2 * beam_size)
        cols = wrd_dist.size(1)
        # we'll break as soon as <eos> is at the top of the beam.
        if top2k[0] == eos_idx and i >= min_len:
            final_hyp = backtrace(curr_hyps[0])
            final_hyp.append(eos_idx)
            return final_hyp, maxprobs[0]

        new_hyps, anc_h1, anc_c1, anc_h2, anc_c2 = [], [], [], [], []
        for k in range(2 * beam_size):
            anc, wrd = top2k[k] / cols, top2k[k] % cols
            curr_scores[len(new_hyps)][0] = maxprobs[k]
            inps[len(new_hyps)] = wrd if wrd < len(i2w) and wrd >= 4 else unk_idx  # the first 4 are symbols

            new_hyps.append((wrd, curr_hyps[anc]))
            anc_h1.append(h1[:, anc, :])
            anc_c1.append(c1[:, anc, :])  # todo: if i+1 is start of begin, change to inits_h1 inits_c1
            anc_h2.append(h2[:, anc, :])
            anc_c2.append(c2[:, anc, :])
            if len(new_hyps) == beam_size:
                break

        assert len(new_hyps) == beam_size
        curr_hyps = new_hyps

        h1 = torch.stack(anc_h1, 1)
        c1 = torch.stack(anc_c1, 1)
        h2 = torch.stack(anc_h2, 1)
        c2 = torch.stack(anc_c2, 1)
        inp_emb = net.hsmm_net.lut(inps).unsqueeze(0)  # 1 x bsz x emb_size
        bsz, src_len, src_emb_size = src_sent_enc.size()
        if bsz < beam_size:
            src_sent_enc = src_sent_enc.expand(beam_size, src_len, src_emb_size)
            src_mask = src_mask.expand(beam_size, src_len)

    # if reach max response len
    final_hyp = backtrace(curr_hyps[0])
    final_hyp.append(eos_idx)
    return final_hyp, curr_scores[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate')
    parser.add_argument('-gen_from_fi', type=str, help='the test source file.')
    parser.add_argument('-ntemplates', type=int, default=100)

    parser.add_argument('-tagged_fi', type=str, help='the tagged template pool file.')
    parser.add_argument('-generator', type=str, help='the generator model.')
    parser.add_argument('-hsmm', type=str, help='the nhsmm model.')

    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-test_src', type=str, default='', help='the test query file.')
    parser.add_argument('-test_rep', type=str, default='', help='the test response file')
    parser.add_argument('-lang', type=str, default='')
    parser.add_argument('-output', type=str, help='the output dir.')

    args = parser.parse_args()

    '''
    python generate.py -gen_from_fi question_data/dia_data/test.post -tagged_fi segs/hsmm-300-50-4.pt.e10.segs.pool -generator models/generator_adversarial.pt.e10 -hsmm models/hsmm-300-50-4.pt.e10 -output output_question'''

    from template_extraction import extract_from_tagged_data
    top_temps, temps2sents, state2phrases = extract_from_tagged_data(args.tagged_fi, args.ntemplates)

    # load hsmm
    saved_stuff = torch.load(args.hsmm)
    vocab = saved_stuff["dict"]
    hsmm = HSMM(len(vocab), len(vocab), saved_stuff["opt"])
    # remove the 'module' prefix
    clean_state = {}
    saved_state = saved_stuff["state_dict"]
    for k, v in saved_state.items():
        nk = k[7:] if k.startswith('module.') else k
        clean_state[nk] = v
    hsmm.load_state_dict(clean_state, strict=True)

    # load generator
    saved_stuff = torch.load(args.generator)
    generator = generator.Generator(hsmm, 300, vocab, 0)
    generator.load_state_dict(saved_stuff['state_dict'])
    generator = generator.cuda()

    with open(args.gen_from_fi) as f:
        src_lines = [_.strip() for _ in f.readlines()]#[:50]
        whole_res = {}

    res = []
    for ll, src_line in enumerate(src_lines):
        query_id = [vocab.add_word(_) for _ in src_line.strip().split()]
        src_sent = torch.LongTensor(query_id).unsqueeze(0).cuda()  # 1 x src_len

        '''sample one template here and generate a response conditioned on the query.'''
        template = generator.template_sample(top_temps[:50],limit=15).cuda()
        rep, score = beam_search(generator, vocab, src_sent, args.beam_size, template)
        query = ' '.join([vocab.idx2word[_] for _ in query_id])
        response = ' '.join([vocab.idx2word[_] for _ in rep])
        tqdm.write(f'{query} ||| {response} ||| {template}')
        res.append(response[:-6] if response.endswith('<eos>') else response)
    with open(os.path.join(args.output, 'gen_response.txt'), 'w', encoding='utf8')as p:
        p.writelines([_ + '\n' for _ in res])
    exit()
