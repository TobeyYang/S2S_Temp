from __future__ import print_function
import os

import sys
import torch
import torch.optim as optim
import torch.nn as nn
from chsmm_without_src import HSMM
import generator
import discriminator
import datasets
import argparse
from tqdm import tqdm
from utils import batchwise_sample
import random
import numpy as np

# from generate import metric_validate

parser = argparse.ArgumentParser('gan args')
parser.add_argument('-data', type=str, default='question_data/dia_parse', help='path to data dir')
parser.add_argument('-bsz', type=int, default=32, help='batch size')
parser.add_argument('-emb_size', type=int, default=300, help='size of word embeddings')
parser.add_argument('-hid_size', type=int, default=300, help='size of rnn hidden state')
parser.add_argument('-dropout', type=float, default=0)
parser.add_argument('-load', type=str, help='the path to load hsmm model')

parser.add_argument('-pretrain_gen', action='store_true', help='pre-train generator by MLE')
parser.add_argument('-pretrain_gen_epochs', type=int, default=20)
parser.add_argument('-pretrain_train_fi', type=str, help='the segmented training data.')
parser.add_argument('-pretrain_valid_fi', type=str, help='the segmented validation data.')
parser.add_argument('-pretrained_gen_path', type=str, default='models/pretrain_gen.pt',
                    help='the save path of pretrained generator.')
parser.add_argument('-load_gen', action='store_true', help='load the pre-trained generator.')
parser.add_argument('-N', type=int, default=5, help='the roll out numbers.')
parser.add_argument('-beam_sample', action='store_true', help='whether use beam search sample strategy')
parser.add_argument('-beam_size', type=int, default=5, help='the beam search size.')
parser.add_argument('-no_teacher', action='store_true', help='no teacher forcing.')

parser.add_argument('-test_src', type=str, default='', help='the test query file.')
parser.add_argument('-test_rep', type=str, default='', help='the test response file')

parser.add_argument('-pretrain_dis', action='store_true', help='pre-train discriminator by neg sample')
parser.add_argument('-pretrain_dis_epochs', type=int, default=20)
parser.add_argument('-pretrained_dis_path', type=str, default='models/pretrain_dis.pt')
parser.add_argument('-load_dis', action='store_true', help='load the pre-trained discriminator.')
parser.add_argument('-pos_neg_samples', type=int, default=20000, help='the positive and negative samples number')
parser.add_argument('--filter_num', type=int, default=128)
parser.add_argument('--filter_sizes', type=str, default='1,2,3')
parser.add_argument('-d_steps', type=int, default=1)
parser.add_argument('-dis_epochs', type=int, default=2)

parser.add_argument('-tagged_fi', type=str, help='the template pool file.')
parser.add_argument('-ntemplates', type=int, default=100)

parser.add_argument('-adv_train_epochs', type=int, default=20, help='the adversarial training epochs.')
parser.add_argument('-adv_gen_batch_num', type=int, default=50)  # todo: increase the number
parser.add_argument('-gan_path', type=str, default='models/gan',
                    help='the directory to save generator and discriminator during adversarial learning.')

# parser.add_argument('-lang', required=True)

parser.add_argument('--gpu', type=str)
parser.add_argument('--dataDir', type=str)
parser.add_argument('--modelDir', type=str)
parser.add_argument('--logDir', type=str)

args = parser.parse_args()
print(vars(args))

'''
python gan.py  -data question_data/dia_data -load models/hsmm-300-50-4.pt.e10 -pretrain_gen -pretrained_gen_path models/pretrain_gen.pt -bsz 100 --gpu 0 -pretrain_train_fi segs/hsmm-300-50-4.pt.e10.segs.train -pretrain_valid_fi segs/hsmm-300-50-4.pt.e10.segs.valid -beam_sample -tagged_fi segs/hsmm-300-50-4.pt.e10.segs.pool -gan_path models/generator_adversarial.pt'''


def validation(gen):
    gen.eval()
    total_loss = []
    with torch.no_grad():
        for i in range(len(corpus.valid)):
            queries, responses, templates = corpus.valid[i]
            queries, responses, templates = queries.cuda(), responses.cuda(), templates.cuda()
            loss = gen.batch_nll_loss(queries, responses, templates)
            total_loss.append(loss)
    # print(gen.decoder.weight.data[0][:20])
    return sum(total_loss) / len(total_loss)


def train_generator_MLE(gen, gen_opt, epochs):
    '''
    Max Likelihood Pretraining for the generator
    Args:
        gen: generator
        gen_opt: optimizer
        epochs:

    Returns:

    '''
    # init val
    tqdm.write(f'init val loss: {validation(gen):.4f}')
    gen.train()
    best_val_loss, best_epoch = float("inf"), 0
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1))
        total_loss = []

        trainperm = torch.randperm(len(corpus.train))
        batch_num = len(corpus.train)
        for bid in range(batch_num):
            queries, responses, tpls = corpus.train[trainperm[bid]]
            queries, responses, tpls = queries.cuda(), responses.cuda(), tpls.cuda()
            loss = gen.batch_nll_loss(queries, responses, tpls)
            # print(loss)
            loss.backward()
            gen_opt.step()

            total_loss.append(loss.data.item())

            if (bid + 1) % 500 == 0:
                tqdm.write(f'pretrain loss: {sum(total_loss[-200:])/200:.4f}')

            if (bid + 1) % (len(corpus.train) // 10) == 0:
                val_loss = validation(gen)
                if val_loss < best_val_loss:
                    best_val_loss, best_epoch = val_loss, epoch
                    kwargs = {'loss': best_val_loss, 'epoch': epoch + 1, 'opt': args}
                    gen.save_model(args.pretrained_gen_path, **kwargs)
                    tqdm.write(f'updated best val loss: {val_loss:.4f}  model saved to {args.pretrained_gen_path}')
                else:
                    tqdm.write(f'valid loss: {val_loss:.4f}')

                gen.train()

        avg_loss = sum(total_loss) / len(total_loss)
        tqdm.write(f'e{epoch+1} average loss: {avg_loss:.4f}')
        kwargs = {'loss': best_val_loss, 'epoch': epoch + 1, 'opt': args}
        path = args.pretrained_gen_path + f'.e{epoch+1}'
        gen.save_model(path, **kwargs)
        print(f'save model to {path}')


def train_generator_PG(gen, gen_opt, dis, batches_num, templates):
    '''
    The generator is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
    Args:
        gen: generator
        gen_opt: optimizer
        dis: discriminator
        batches_num: batch number.
        templates: list of templates.
    Returns:

    '''
    gen.train()
    perm = list(range(len(corpus.train)))
    random.shuffle(perm)
    samples_ids = perm[:batches_num]
    for bid in samples_ids:
        query, rep, template = corpus.train[bid]
        query, rep, template = query.cuda(), rep.cuda(), template.cuda()
        bsz, tpl_len = template.size(0), 15

        # pg loss
        tpls = torch.stack([gen.template_sample(templates, tpl_len) for _ in range(bsz)], 0).cuda()
        if args.beam_sample:
            sample_rep = gen.beam_sample(query, tpls, args.beam_size)
        else:
            sample_rep = gen.sample_one(query, None, tpls)  # the sampled response

        out = dis.batchClassify(query, sample_rep)
        # roll out
        rewards = torch.zeros(bsz, tpl_len).cuda()
        for i in range(tpl_len):
            rw = torch.zeros(bsz).cuda()
            for n in range(args.N):  # roll out N times
                mc = gen.sample_one(query, sample_rep[:, :i + 1], tpls)  # sample the unknown tpl_len-i tokens
                rw += dis.forward(query, mc)  # get the rewards
            rewards[:, i] = rw / args.N
        pg_loss = gen.batch_reward_loss(query, sample_rep, tpls, rewards)
        gen_opt.zero_grad()
        pg_loss.backward()
        gen_opt.step()

        # teacher forcing
        if not args.no_teacher:
            loss = gen.batch_nll_loss(query, rep, template)
            gen_opt.zero_grad()
            loss.backward()
            gen_opt.step()


def train_discriminator(dis, dis_opt, gen, d_steps, epochs, templates, pos_neg_samples):
    """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
        """
    # get valid data
    best_val_acc = 0
    for d_step in range(d_steps):
        # pos_neg_samples = 200
        data = batchwise_sample(corpus.train, gen, pos_neg_samples // args.bsz, templates, max_rep_len=15)
        valid_data = batchwise_sample(corpus.valid, gen, len(corpus.valid), templates, max_rep_len=15, is_train=False)
        for e in range(epochs):
            total_loss = 0
            # for i in tqdm(range(len(data)), desc=f'd{d_step+1}_e{e+1}', leave=False):
            for i in range(len(data)):
                src, rep, target = data[i]
                dis_opt.zero_grad()
                loss = dis.batchBCEloss(src, rep, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data)

            val_acc = dis.validate(valid_data)
            tqdm.write(f'd{d_step+1}_e{e+1}  avg loss: {avg_loss:.4f}  val acc: {val_acc:.4f}')
            if val_acc > best_val_acc:
                spath = os.path.join(args.gan_path, 'dis')
                dis.save_model(spath + f'.s{d_step}.e{e}')
                best_val_acc = val_acc


def train_discriminator_matching(dis, dis_opt, epochs):
    matching_corpus = datasets.MatchingCorpus(os.path.join(args.data, 'matching_train.txt'),
                                              os.path.join(args.data, 'matching_valid.txt'),
                                              args.bsz, vocab)
    train_data = matching_corpus.train
    valid_data = matching_corpus.valid
    val_acc = dis.validate(valid_data)
    print(val_acc)
    best_acc = val_acc
    for e in range(epochs):
        total_loss = 0
        for i in tqdm(range(len(train_data))):
            query, rep, target = train_data[i]
            # query, rep, target = query.cuda(), rep.cuda(), target.cuda()
            dis_opt.zero_grad()
            loss = dis.batchBCEloss(query, rep, target)
            loss.backward()
            dis_opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)

        val_acc = dis.validate(valid_data)
        tqdm.write(f'epoch: {e+1}  avg loss: {avg_loss:.4f} val acc: {val_acc:.4f}')
        if val_acc > best_acc:
            path = args.pretrained_dis_path + f'.match'
            dis.save_model(path)
            tqdm.write(f'save matching model to {path}')
            best_acc = val_acc


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)

    os.makedirs(args.gan_path, exist_ok=True)

    saved_stuff = torch.load(args.load)
    vocab = saved_stuff["dict"]
    corpus = datasets.TemplateCorpus(args.pretrain_train_fi, args.pretrain_valid_fi, args.bsz, vocab)

    net = HSMM(len(vocab), len(vocab), saved_stuff["opt"])
    net.load_model(args.load)

    from template_extraction import extract_from_tagged_data

    templates, _, _ = extract_from_tagged_data(args.tagged_fi, args.ntemplates)

    gen = generator.Generator(net, args.hid_size, vocab, args.dropout)
    dis = discriminator.Discriminator(len(vocab), args.emb_size, args.filter_num, eval(args.filter_sizes), args.dropout)
    gen = gen.cuda()
    dis = dis.cuda()

    gen_opt = optim.Adam(filter(lambda p: p.requires_grad, gen.parameters()), lr=1e-5)

    if args.load_gen:
        gen.load_model(args.pretrained_gen_path)
        print(f'generator loaded from {args.pretrained_gen_path}')
    if args.pretrain_gen:
        print('Starting Generator MLE Training...')
        train_generator_MLE(gen, gen_opt, args.pretrain_gen_epochs)

    # pre-train discriminator
    dis_opt = optim.Adam(dis.parameters(), lr=1e-3)
    if args.load_dis:
        path = args.pretrained_dis_path + '.match'
        dis.load_model(path)
        print(f'discriminator loaded from {path}')
    if args.pretrain_dis:
        print('Pretrain Discriminator by negative sampling')
        train_discriminator_matching(dis, dis_opt, args.pretrain_dis_epochs)

    train_discriminator(dis, dis_opt, gen, args.d_steps, args.dis_epochs, templates, args.pos_neg_samples)
    # Adversarial Training
    print('Starting Adversarial Training...')
    print(f'Init validation loss: {validation(gen):.4f}')


    for epoch in range(1, args.adv_train_epochs + 1):
        print('\n--------\nEPOCH %d\n--------' % (epoch))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ')
        train_generator_PG(gen, gen_opt, dis, args.adv_gen_batch_num, templates)
        kwargs = {'epoch': epoch, 'opt': args}
        path = os.path.join(args.gan_path, f'gen.e{epoch}')
        gen.save_model(path, **kwargs)
        print(f'save model to {path}')

        print(f'epoch: {epoch}  val_nll_loss: {validation(gen):.4f}')

        # Train Discriminator
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_opt, gen, args.d_steps, args.dis_epochs, templates, args.pos_neg_samples)
