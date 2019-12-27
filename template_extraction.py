import re
from collections import defaultdict

import torch

# import labeled_data
import datasets

seg_patt = re.compile('([^\|]+)\|(\d+)') # detects segments

def group_by_template(fi, startlineno):
    """
    returns a label-tup -> [(phrase-list, lineno), ...] map
    """
    labes2sents = defaultdict(list)
    lineno = startlineno
    with open(fi, encoding='utf8') as f:
        for line in f:
            if '|' not in line:
                continue
            if '|||' in line:
                line = line.split('|||')[1]
            seq = seg_patt.findall(line.strip()) # list of 2-tuples
            wordseq, labeseq = zip(*seq) # 2 tuples
            wordseq = [phrs.strip() for phrs in wordseq]
            labeseq = tuple(int(labe) for labe in labeseq)
            labes2sents[labeseq].append((wordseq, lineno))
            lineno += 1
            # if lineno > 1000:
            #     break
    return labes2sents

def remap_eos_states(top_temps, temps2sents):
    """
    allocates a new state for any state that is also used for an <eos>
    """
    used_states = set()
    [used_states.update(temp) for temp in top_temps]
    final_states = set()
    for temp in top_temps:
        final_state = temp[-1]
        assert any(sent[-1] == "<eos>" for sent, lineno in temps2sents[temp])
        final_states.add(final_state)

    # make new states
    remap = {}
    for i, temp in enumerate(top_temps):
        nutemp = []
        changed = False
        for j, t in enumerate(temp):
            if j < len(temp)-1 and t in final_states:
                changed = True
                if t not in remap:
                    remap[t] = max(used_states) + len(remap) + 1
            nutemp.append(remap[t] if t in remap else t)
        if changed:
            nutuple = tuple(nutemp)
            top_temps[i] = nutuple
            temps2sents[nutuple] = temps2sents[temp]
            del temps2sents[temp]

def just_state2phrases(temps, temps2sents):
    state2phrases = defaultdict(lambda: defaultdict(int)) # defaultdict of defaultdict
    for temp in temps:
        for sent, lineno in temps2sents[temp]:
            for i, state in enumerate(temp):
                #state2phrases[state].add(sent[i])
                state2phrases[state][sent[i]] += 1

    nustate2phrases = {}
    for k, v in state2phrases.items():
        phrases = list(v)
        counts = torch.Tensor([state2phrases[k][phrs] for phrs in phrases])
        counts.div_(counts.sum())
        nustate2phrases[k] = (phrases, counts)
    state2phrases = nustate2phrases
    return state2phrases


def extract_from_tagged_data(tagged_fi, ntemplates):
    nskips = 0
    temps2sents = group_by_template(tagged_fi, nskips)
    top_temps = sorted(temps2sents.keys(), key=lambda x: -len(temps2sents[x]))
    #temp length larger than 3.
    top_temps = [_ for _ in top_temps if len(_)>3][:ntemplates]


    #remap_eos_states(top_temps, temps2sents)
    state2phrases = just_state2phrases(top_temps, temps2sents)
    return top_temps, temps2sents, state2phrases


def topk_phrases(pobj, k):
    phrases, probs = pobj
    thing = sorted(zip(phrases, list(probs)), key=lambda x: -x[1])
    sphrases, sprobs = zip(*thing)
    return sphrases[:k]


def align_cntr(cntr, thresh=0.4):
    tote = float(sum(cntr.values()))
    nug = {k : v/tote for k, v in cntr.items()}
    best, bestp = None, 0
    for k, v in nug.items():
        if v > bestp:
            best, bestp = k, v
    if bestp >= thresh:
        return best
    else:
        return None
