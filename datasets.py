"""
this file modified from the word_language_model example
"""
import os
import torch
from codecs import open
from collections import Counter, defaultdict
import re

import random

random.seed(1111)

# punctuation = set(['.', '!', ',', ';', ':', '?', '--', '-rrb-', '-lrb-'])
punctuation = set()  # i don't know why i was so worried about punctuation


class Dictionary(object):
    def __init__(self, unk_word="<unk>"):
        self.unk_word = unk_word
        self.idx2word = [unk_word, "<pad>", "<bos>", "<eos>", "<post>"]  # OpenNMT constants
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def add_word(self, word, train=False):
        """
        returns idx of word
        """
        if train and word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word] if word in self.word2idx else self.word2idx[self.unk_word]

    def bulk_add(self, words):
        """
        assumes train=True
        """
        self.idx2word.extend(words)
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def load_from_file(self, vocab_file, size=None):
        with open(vocab_file, 'r', encoding='utf8')as p:
            for i, line in enumerate(p):
                if i > size:
                    break
                word = line.strip().split()[0]
                self.add_word(word, True)
        print("load vocab from {}, vocab size: {}".format(vocab_file, len(self.idx2word)))

    def __len__(self):
        return len(self.idx2word)


class MaskDictionary(object):
    def __init__(self, unk_word="<unk>"):
        self.unk_word = unk_word
        self.idx2word = [unk_word, "<pad>", "<bos>", "<eos>", "<post>"]  # OpenNMT constants
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}

    def add_word(self, word, train=False):
        """
        returns idx of word
        """
        if train and word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word] if word in self.word2idx else self.word2idx[self.unk_word]

    def load_from_file(self, interrogative_file, ordinary_file, topic_file, size=None):
        inter_words, ordinary_words, topic_words = set(), set(), set()
        ordinary_words.update([self.unk_word, "<pad>", "<bos>", "<eos>", "<post>"])
        ordinary_size = 250

        with open(interrogative_file, encoding='utf8')as p:
            for i, line in enumerate(p):
                if len(self) >= size:
                    break
                word = line.strip().split()[0]
                self.add_word(word, True)
                inter_words.add(word)

        with open(ordinary_file, 'r', encoding='utf8')as p:
            for i, line in enumerate(p):
                if len(self) >= size or i >= ordinary_size:
                    break
                word = line.strip().split()[0]
                self.add_word(word, True)
                ordinary_words.add(word)

        with open(topic_file, encoding='utf8')as p:
            for i, line in enumerate(p):
                if len(self) >= size:
                    break
                word = line.strip().split()[0]
                self.add_word(word, True)
                topic_words.add(word)

        print("load vocab with size: {}".format(len(self.idx2word)))
        print("interrogative size: {}".format(len(inter_words)))
        print("topic size: {}".format(len(topic_words)))
        print("ordinary size: {}".format(len(ordinary_words)))

        # w = 1e-6
        w = 0
        self.interrogative_mask = [w if _ not in inter_words else 1 for _ in self.idx2word]
        self.ordinary_mask = [w if _ not in ordinary_words else 1 for _ in self.idx2word]
        self.topic_mask = [w if _ not in topic_words else 1 for _ in self.idx2word]

        # check the masks
        import numpy as np
        iter_mask, ord_mask, top_mask = np.array(self.interrogative_mask), np.array(self.ordinary_mask), np.array(
            self.topic_mask)
        mask = iter_mask + ord_mask + top_mask
        print("other size: {}".format(sum(mask < 1)))
        pass

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, bsz, vocab, add_bos=False, add_eos=False, test=False, quick=False):
        self.dictionary = vocab
        self.bsz = bsz

        train_src = os.path.join(path, 'src_train.txt')
        train_tgt = os.path.join(path, 'train.txt')
        valid_src = os.path.join(path, 'src_valid.txt')
        valid_tgt = os.path.join(path, 'valid.txt')
        test_src = os.path.join(path, 'src_test.txt')
        test_tgt = os.path.join(path, 'test.txt')

        self.ngen_types = len(vocab)
        if not quick:
            trsrc_sents, trsents, trlabels, trinps = self.tokenize(train_tgt, train_src, add_bos, add_eos)

            self.train, self.train_mb2linenos = self.minibatchify(trsrc_sents, trsents, trlabels, trinps, self.bsz)
            print("train size: {}  batch num: {}".format(len(trsrc_sents), len(self.train)))

            if os.path.exists(valid_src) or os.path.exists(test_src):
                if not test:
                    vsrc_sents, vsents, vlabels, vinps = self.tokenize(valid_tgt, valid_src, add_bos, add_eos)

                else:
                    print("using test data to valid...")
                    vsrc_sents, vsents, vlabels, vinps = self.tokenize(test_tgt, test_src, add_bos, add_eos)

                self.valid, self.val_mb2linenos = self.minibatchify(vsrc_sents, vsents, vlabels, vinps, self.bsz)
                print("valid size: {}  batch num: {}".format(len(vsrc_sents), len(self.valid)))

    def tokenize(self, path, src_path, add_bos=False, add_eos=False):
        assert os.path.exists(path), "path error: {}".format(path)
        assert os.path.exists(src_path), "src_path error: {}".format(src_path)

        w2i = self.dictionary.word2idx
        src_sents, sents, labels, inps = [], [], [], []
        with open(src_path, 'r', encoding='utf8')as f, open(path, 'r', encoding='utf8')as p:
            for src_line, rp_line in zip(f, p):
                try:
                    tokens = src_line.strip().split()
                    src_sent = [self.dictionary.add_word(_) for _ in tokens]

                    sent, insent = [], []
                    tokens, span_labels = rp_line.strip().split('|||')
                    tokens = tokens.split()
                    # remove the <eos>
                    # tokens = tokens[:-1]

                    if add_bos:
                        sent.append(w2i['<bos>'])
                    sent += [self.dictionary.add_word(_) for _ in tokens]
                    insent += [self.dictionary.add_word(_) for _ in tokens]
                    if add_eos:
                        sent.append(w2i['<eos>'])

                    labetups = [tupstr.split(',') for tupstr in span_labels.split()]
                    labelist = [(int(tup[0]), int(tup[1]), int(tup[2])) for tup in labetups]
                    labelist = [_ for _ in labelist if _[1] - _[0] <= 4]  # remove label over more than 4 tokens.

                    # remove the pair with unk more than 30%
                    if len([1 for _ in sent if _ == w2i["<unk>"]]) / len(sent) > 0.6:
                        # print(rp_line)
                        continue
                    if len([1 for _ in src_sent if _ == w2i["<unk>"]]) / len(src_sent) > 0.6:
                        # print(src_line.strip() + '\t' + rp_line)
                        continue

                    src_sents.append(src_sent)
                    labels.append(labelist)
                    sents.append(sent)
                    inps.append(insent)
                except:
                    print(rp_line)
                    continue

        assert len(src_sents) == len(sents)
        assert len(inps) == len(sents)
        return src_sents, sents, labels, inps

    def padded_src_sents(self, src_sents, pad_id):
        """

        :param src_sents:
        :return: bsz x src_seq_len
        """
        max_len = max([len(_) for _ in src_sents])
        for sent in src_sents:
            while len(sent) < max_len:
                sent.append(pad_id)
        return torch.LongTensor(src_sents)

    def minibatchify(self, src_sents, sents, labels, inps, bsz):
        """
        this should result in there never being any padding.
        each minibatch is:
          (seqlen x bsz, bsz-length list of lists of (start, end, label) constraints,
           bsz x nfields x nfeats, seqlen x bsz x max_locs, seqlen x bsz x max_locs x nfeats)
        """
        # sort in ascending order
        sents, sorted_idxs = zip(*sorted(zip(sents, range(len(sents))), key=lambda x: len(x[0])))
        minibatches, mb2linenos = [], []
        curr_batch, curr_src_sents, curr_labels, curr_inps, curr_linenos = [], [], [], [], []
        curr_len = len(sents[0])

        pad_id = self.dictionary.word2idx['<pad>']
        for i in range(len(sents)):
            if len(sents[i]) != curr_len or len(curr_batch) == bsz:  # we're done
                minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                    curr_labels,
                                    self.padded_src_sents(curr_src_sents, pad_id),
                                    torch.LongTensor(curr_inps).t().contiguous()))
                mb2linenos.append(curr_linenos)
                curr_batch = [sents[i]]
                curr_len = len(sents[i])
                curr_src_sents = [src_sents[sorted_idxs[i]]]
                curr_labels = [labels[sorted_idxs[i]]]
                curr_inps = [inps[sorted_idxs[i]]]
                curr_linenos = [sorted_idxs[i]]
            else:
                curr_batch.append(sents[i])
                curr_src_sents.append(src_sents[sorted_idxs[i]])
                curr_labels.append(labels[sorted_idxs[i]])
                curr_inps.append(inps[sorted_idxs[i]])
                curr_linenos.append(sorted_idxs[i])
        # catch last
        if len(curr_batch) > 0:
            minibatches.append((torch.LongTensor(curr_batch).t().contiguous(),
                                curr_labels,
                                self.padded_src_sents(curr_src_sents, pad_id),
                                torch.LongTensor(curr_inps).transpose(0, 1).contiguous()))
            mb2linenos.append(curr_linenos)
        return minibatches, mb2linenos


class TemplateCorpus(object):
    def __init__(self, train_fi, eval_fi, bsz, vocab):
        self.dictionary = vocab
        self.bsz = bsz
        self.train_fi = train_fi
        self.eval_fi = eval_fi

        if train_fi:
            tr_queries, tr_responses, tr_templates = self.tokenize(self.train_fi)
            self.train = self.minibatchify(tr_queries, tr_responses, tr_templates)
            print(f'train size: {len(tr_queries)} batch num: {len(self.train)}')

        if eval_fi:
            self.vl_queries, self.vl_responses, self.vl_templates = self.tokenize(self.eval_fi)
            self.valid = self.minibatchify(self.vl_queries, self.vl_responses, self.vl_templates)
            print(f'valid size: {len(self.vl_queries)} batch num: {len(self.valid)}')

    def tokenize(self, fi):
        w2i = self.dictionary.word2idx
        queries, responses, templates, seg_lens = [], [], [], []
        with open(fi, encoding='utf8')as p:
            for line in p:
                if '|||' not in line or len(line.split('|||'))!=2:
                    continue  # skip top lines.


                raw_query, raw_rep = line.split('|||')
                query = [w2i[_] for _ in raw_query.strip().split()]
                response, template = [], []
                end = [-1]  # record the split point.
                for i, t in enumerate(raw_rep.strip().split()):
                    m = re.match('(.+)\|(\w+)', t)
                    if m:
                        wrd, tpl = m.group(1), int(m.group(2))
                        response.append(w2i[wrd])
                        seg_len = i - end[-1]
                        end.append(i)
                        template += [tpl] * seg_len
                    else:
                        response.append(w2i[t])

                assert len(template) == len(response)
                queries.append(query)
                responses.append(response)
                templates.append(template)

                # if len(queries) > 200:
                #     break

        return queries, responses, templates

    def padded_queries(self, queries, pad_id, pad_len=3):
        """

        :param queries:
        :return: bsz x src_seq_len
        """
        max_len = max([pad_len] + [len(_) for _ in queries])
        for q in queries:
            while len(q) < max_len:
                q.append(pad_id)
        return queries

    def minibatchify(self, queries, responses, templates):
        responses, queries, templates = zip(
            *sorted(zip(responses, queries, templates), key=lambda x: len(x[0]) * 1e6 + len(x[1])))
        pad_id = self.dictionary.word2idx['<pad>']

        minibatches = []
        curr_query_batch, curr_response_batch, curr_template_batch = [], [], []
        curr_len = len(responses[0])
        for i in range(len(responses)):
            if len(responses[i]) != curr_len or len(curr_query_batch) == self.bsz:
                pad_query = self.padded_queries(curr_query_batch, pad_id)
                minibatches.append((torch.LongTensor(pad_query),
                                    torch.LongTensor(curr_response_batch),
                                    torch.LongTensor(curr_template_batch)))
                curr_len = len(responses[i])
                curr_query_batch = [queries[i]]
                curr_response_batch = [responses[i]]
                curr_template_batch = [templates[i]]

            else:
                curr_query_batch.append(queries[i])
                curr_response_batch.append(responses[i])
                curr_template_batch.append(templates[i])

        # catch the last
        if len(curr_query_batch) > 0:
            pad_query = self.padded_queries(curr_query_batch, pad_id)
            minibatches.append((torch.LongTensor(pad_query),
                                torch.LongTensor(curr_response_batch),
                                torch.LongTensor(curr_template_batch)))

        return minibatches


class MatchingCorpus(object):
    def __init__(self, train_fi, eval_fi, bsz, vocab):
        self.dictionary = vocab
        self.bsz = bsz
        self.train_fi = train_fi
        self.eval_fi = eval_fi

        if train_fi:
            tr_queries, tr_responses, tr_targets = self.tokenize(self.train_fi)
            self.train = self.minibatchify(tr_queries, tr_responses, tr_targets)
            print(f'train size: {len(tr_queries)} batch num: {len(self.train)}')

        if eval_fi:
            self.vl_queries, self.vl_responses, self.vl_targets = self.tokenize(self.eval_fi)
            self.valid = self.minibatchify(self.vl_queries, self.vl_responses, self.vl_targets)
            print(f'valid size: {len(self.vl_queries)} batch num: {len(self.valid)}')

    def tokenize(self, fi):
        queries, reps, targets = [], [], []
        w2i = self.dictionary.word2idx
        with open(fi, encoding='utf8')as p:
            for i ,l in enumerate(p):
                ss = l.strip().split('|||')
                query = ss[0].strip().split()
                rep = ss[1].strip().split()
                target = int(ss[2].strip())

                q_ids = [self.dictionary.add_word(_) for _ in query]
                r_ids = [self.dictionary.add_word(_) for _ in rep]

                if len([1 for _ in q_ids if _ == w2i["<unk>"]]) / len(q_ids) > 0.3:
                    continue
                if len([1 for _ in r_ids if _ == w2i["<unk>"]]) / len(r_ids) > 0.3:
                    continue

                queries.append(q_ids)
                reps.append(r_ids)
                targets.append(target)
                # if i > 2000:
                #     break
        return queries, reps, targets

    def padded_batch(self, queries, pad_id, pad_len=5):
        """

        :param queries:
        :return: bsz x src_seq_len
        """
        max_len = max([pad_len] + [len(_) for _ in queries])
        for q in queries:
            while len(q) < max_len:
                q.append(pad_id)
        return queries

    def minibatchify(self, queries, responses, targets):
        responses, queries, targets = zip(
            *sorted(zip(responses, queries, targets), key=lambda x: len(x[0]) * 1e6 + len(x[1])))
        pad_id = self.dictionary.word2idx['<pad>']

        minibatches = []
        curr_query_batch, curr_response_batch, curr_target_batch = [], [], []
        curr_len = len(responses[0])
        for i in range(len(responses)):
            if len(responses[i]) != curr_len or len(curr_query_batch) == self.bsz:
                pad_query = self.padded_batch(curr_query_batch, pad_id)
                pad_rep = self.padded_batch(curr_response_batch, pad_id, 15)
                minibatches.append((torch.LongTensor(pad_query).cuda(),
                                    torch.LongTensor(pad_rep).cuda(),
                                    torch.Tensor(curr_target_batch).cuda()))
                curr_len = len(responses[i])
                curr_query_batch = [queries[i]]
                curr_response_batch = [responses[i]]
                curr_target_batch = [targets[i]]

            else:
                curr_query_batch.append(queries[i])
                curr_response_batch.append(responses[i])
                curr_target_batch.append(targets[i])

        # catch the last
        if len(curr_query_batch) > 0:
            pad_query = self.padded_batch(curr_query_batch, pad_id)
            pad_rep = self.padded_batch(curr_response_batch, pad_id, 15)
            minibatches.append((torch.LongTensor(pad_query).cuda(),
                                torch.LongTensor(pad_rep).cuda(),
                                torch.Tensor(curr_target_batch).cuda()))

        return minibatches



if __name__ == '__main__':
    pass
