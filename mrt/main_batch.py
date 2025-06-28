#!/usr/bin/env python
# coding:utf8

import argparse
import os
import numpy
import torch
import time
from preprocess_data import getdata
import random

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from nnet.btransform import BTransfrom_MRT
from tutils import test_detail, test, train, append_log_test, append_log_train_valid
from quick_test import quick_test

os.environ['CUDA_LAUNCH_BLOCKING'] = '5'


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # define location to save the model
    set_seed(args.seed)
    if args.save == "__":
        args.save = "model/%s_len%d_batch%d_%s" % \
                    (args.model, args.sen_max_len, args.batch_size, args.tag)

    if args.is_test:
        print("Begin to preprocess only Test Pairs")
        tic = time.time()
        test_set, word_size = getdata(args.data, args.train_pair, args.valid_pair, args.test_pair, args.dictionary
                                                                 , args.ast_type, args.sen_max_len, quick_test=True)
        #print("Quick Test Preprocess using %.6f seconds" % (time.time() - tic))
        print("Begin to test")    
        tic = time.time()
        model = torch.load(args.save + "/model.pt")
        if args.quick_test:
            quick_test(model, test_set, args, to_save=True, threshold=args.threshold)
        else:
            test_detail(model, test_set, args, to_save=True)
        print("Quick Test using %.6f seconds" % (time.time() - tic))
    else:
        tic = time.time()
        [training_set, valid_set, test_set], word_size = getdata(args.data, args.train_pair, args.valid_pair,
                                                                 args.test_pair, args.dictionary
                                                                 , args.ast_type, args.sen_max_len)
        #print("Normal Preprocess using %.6f seconds" % (time.time() - tic))

        word_size = word_size + 1
        dt_string = datetime.now().strftime("%m_%d_%H_%M_%S")
        dt_string = dt_string + f'_lr={args.lr}'
        writer = SummaryWriter(log_dir=args.save + '/log_' + dt_string)

        if not os.path.exists(args.save):
            os.mkdir(args.save)

        models = {
            "BTransfrom_MRT": BTransfrom_MRT
        }
        model = models[args.model](max_len=args.sen_max_len, num_embeddings=word_size,
                                   N=args.transformer_nlayers, d_model=args.d_model,
                                   d_ff=args.d_ff, h=args.h, dropout=args.dropout, output_dim=args.output_dim)

        if torch.cuda.is_available():
            if not args.cuda:
                print("Waring: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(args.seed)
                model.cuda()

        best_f1_test, best_f1_valid = -numpy.inf, -numpy.inf
        batches_per_epoch = int(len(training_set) / args.batch_size)
        max_train_steps = int(args.epochs * batches_per_epoch)
        tic = time.time()
        total_loss = 0

        warmup_steps = batches_per_epoch * (args.epochs / 5)
        target_lr = args.lr
        final_lr = args.lr / 10

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return target_lr * current_step / warmup_steps
            else:
                return max(final_lr, target_lr - (target_lr - final_lr) * (current_step - warmup_steps) / (
                            max_train_steps - warmup_steps))

        optimizer = optim.Adam(model.parameters(), lr=1.0)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        criterion = nn.BCELoss()

        tqdm_steps = tqdm(range(max_train_steps))
        for step in tqdm_steps:
            training_batch = training_set.next_batch(args.batch_size)

            current_loss = train(model, training_batch, args, optimizer, criterion)
            total_loss += current_loss
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            tqdm_steps.set_postfix(loss='{:.6f}'.format(current_loss), lr='{:.6f}'.format(current_lr))
            
            if args.valid_step > 0:
                if (step + 1) % args.valid_step == 0:
                    acc_score, f1, test_loss = quick_test(model, valid_set, args, False, args.valid_threshold)
                    print(f"  ----avg loss is {total_loss / args.valid_step}")
                    loss = total_loss / args.valid_step
                    append_log_train_valid(loss, acc_score, f1, test_loss, (step + 1) / args.valid_step, writer)
                    total_loss = 0
                    print(f"STEP{(step + 1)}  ----Old best acc score on [validation] is {best_f1_valid}")
                    if f1 > best_f1_valid:
                        print(f"STEP{(step + 1)}  ----New acc score on [validation] is {f1}")
                        best_f1_valid = f1
                        with open(args.save + "/model.pt", 'wb') as to_save:
                            torch.save(model, to_save)
                        acc_test, f1, test_loss = quick_test(model, test_set, args, False, args.valid_threshold)
                        append_log_test(acc_test, f1, (step + 1) / args.valid_step, writer)
                        print(f"STEP{(step + 1)}  ----Old best acc score on [test] is {best_f1_test}")
                        if acc_test > best_f1_test:
                            best_f1_test = f1
                            print(f"STEP{(step + 1)}  ----New acc score on [test] is {f1}")
            else:
                if (step + 1) % batches_per_epoch == 0:
                    print("One batch using %.5f seconds" % (time.time() - tic))
                    tic = time.time()
                    ''' Test after each epoch '''
                    acc_score, f1, test_loss = quick_test(model, valid_set, args, False, args.valid_threshold)

                    print("  ----avg loss is %f" % (total_loss / batches_per_epoch))
                    loss = total_loss / batches_per_epoch
                    append_log_train_valid(loss, acc_score, f1, test_loss, training_set.epochs_completed, writer)
                    total_loss = 0
                    print("  ----Old best acc score on [validation] is %f" % best_f1_valid)
                    if f1 > best_f1_valid:
                        print("  ----New acc score on [validation] is %f" % f1)
                        best_f1_valid = f1
                        with open(args.save + "/model.pt", 'wb') as to_save:
                            torch.save(model, to_save)
                        acc_test, f1, test_loss = quick_test(model, test_set, args, False, args.valid_threshold)
                        append_log_test(acc_test, f1, training_set.epochs_completed, writer)
                        print("  ----Old best acc score on [test] is %f" % best_f1_test)
                        if acc_test > best_f1_test:
                            best_f1_test = f1
                            print("  ----New acc score on [test] is %f" % f1)

        model = torch.load(args.save + "/model.pt")
        test_detail(model, test_set, args, to_save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch AoA for Stance Project")

    ''' load data and save model'''
    parser.add_argument("--save", type=str, default="__",
                        help="path to save the model")
    parser.add_argument("--tag", type=str, default="GCJ_OAST",
                        help="tag of dataset")
    parser.add_argument("--data", type=str, default="origindata/GCJ_with_AST+OAST.csv",
                        help="path to dataset")
    parser.add_argument("--train_pair", type=str, default="origindata/GCJ_train11.csv",
                        help="path to train dataset")
    parser.add_argument("--test_pair", type=str, default="origindata/GCJ_test.csv",
                        help="path to test dataset")
    parser.add_argument("--valid_pair", type=str, default="origindata/GCJ_valid.csv",
                        help="path to valid dataset")
    parser.add_argument("--dictionary", type=str, default="origindata/GCJ_XXX_dictionary.txt",
                        help="path to dictionary")
    parser.add_argument("--ast_type", type=str, default="OAST",
                        help="AST Type")

    ''' training parameters '''
    parser.add_argument("--model", type=str, default="BTransfrom_MRT",
                        help="type of model")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epoch")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout rate")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="gamma")
    parser.add_argument("--seed", type=int, default=1334,
                        help="random seed for reproduction")
    parser.add_argument("--cuda", action="store_true",
                        help="use CUDA")

    ''' model parameters '''
    parser.add_argument("--sen_max_len", type=int, default=256,
                        help="max sequence length")
    parser.add_argument("--transformer_nlayers", type=int, default=2,
                        help="num of layers of transformer")
    parser.add_argument("--d_model", type=int, default=128,
                        help="dim of the attention layer")
    parser.add_argument("--d_ff", type=int, default=512,
                        help="d_ff of the attention layer")
    parser.add_argument("--h", type=int, default=8,
                        help="num of heads of attention layer")
    parser.add_argument("--output_dim", type=int, default=512,
                        help="dim of output feature")

    ''' test and valid purpose'''
    parser.add_argument("--is_test", action="store_true",
                        help="flag for training model or only test")
    parser.add_argument("--quick_test", action="store_true",
                        help="flag for using quick test")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="The threshold in test")
    parser.add_argument("--valid_threshold", type=int, default=0.8,
                        help="The threshold in only test")
    parser.add_argument("--valid_step", type=int, default=1750,
                        help="The valid step in train")

    my_args = parser.parse_args()
    torch.manual_seed(my_args.seed)
    main(my_args)
