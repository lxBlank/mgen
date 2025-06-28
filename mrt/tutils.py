import argparse
import os
import numpy
import math
import torch
import time
import csv
import shutil

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def cal_prf(pred, right, gold, formation=True, metric_type=""):
    num_class = len(pred)
    precision = [0.0] * num_class
    recall = [0.0] * num_class
    f1_score = [0.0] * num_class

    for i in range(num_class):
        precision[i] = 0 if pred[i] == 0 else 1.0 * right[i] / pred[i]
        recall[i] = 0 if gold[i] == 0 else 1.0 * right[i] / gold[i]
        f1_score[i] = 0 if precision[i] == 0 or recall[i] == 0 \
            else 2.0 * (precision[i] * recall[i]) / (precision[i] + recall[i])

        if formation:
            precision[i] = precision[i].__format__(".6f")
            recall[i] = recall[i].__format__(".6f")
            f1_score[i] = f1_score[i].__format__(".6f")

    ''' PRF for each label or PRF for all labels '''
    if metric_type == "macro":
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    elif metric_type == "micro":
        precision = 1.0 * sum(right) / sum(pred) if sum(pred) > 0 else 0
        recall = 1.0 * sum(right) / sum(gold) if sum(recall) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def test_prf(pred, labels):
    total = len(labels)
    pred_right = [0, 0]
    pred_all = [0, 0]
    gold = [0, 0]
    for i in range(total):
        pred_all[pred[i]] += 1
        if pred[i] == labels[i]:
            pred_right[pred[i]] += 1
        gold[labels[i]] += 1

    print(f"Prediction:{pred_all} | Right:{pred_right} | Gold:{gold}")
    accuracy = 1.0 * sum(pred_right) / total
    p, r, f1 = cal_prf(pred_all, pred_right, gold, formation=False)
    _, _, macro_f1 = cal_prf(pred_all, pred_right, gold,
                                    formation=False,
                                    metric_type="macro")
    print(f"Accuracy: {sum(pred_right)}/{total} = {accuracy:.4f} | Precision:{p[1]:.4f} | Recall:{r[1]:.4f} | F1 score:{f1[1]:.4f}")
    return accuracy, f1[1]

CPP_FILE = "../code_clone_preprocess/"

def test_prf_detail(pred, labels, origin_pred, pair_data, origin_data, args, max_wrong = 100, to_save = False):
    if os.path.exists("wrong"):
        shutil.rmtree("wrong")
    os.mkdir("wrong")
    os.mkdir("wrong/1to0")
    os.mkdir("wrong/0to1")
    wrong = 0
    total = len(labels)
    pred_right = [0, 0]
    pred_all = [0, 0]
    gold = [0, 0]
    wfdict = {}
    tdict = {}
    wrong_names = []
    all_names = []
    all_pred = []
    all_gd = []
    all_right = []
    wrong_list_detail = []
    for i in range(total):
        pair_info = pair_data[i]
        index1 = pair_info["index1"]
        index2 = pair_info["index2"]
        file1 = pair_info["file1"]
        file2 = pair_info["file2"]
        file1n = f"{index1}-{file1}"
        file2n = f"{index2}-{file2}"
        fln = file1n + "|" + file2n
        pred_all[pred[i]] += 1
        all_names.append(fln)
        all_pred.append(origin_pred[i])
        all_gd.append(labels[i])
        if pred[i] == labels[i]:
            all_right.append("True")
            pred_right[pred[i]] += 1
        else:
            all_right.append("False")
            if wrong < max_wrong:
                wrong += 1
                wrong_names.append(fln)
                if 0 == labels[i]:
                    f = open(f'wrong/0to1/wrong{file1n}={file2n}.txt', "w")
                else:
                    f = open(f'wrong/1to0/wrong{file1n}={file2n}.txt', "w")
                code1 = origin_data[index1][file1]["code"]
                code2 = origin_data[index2][file2]["code"]
                len1 = origin_data[index1][file1]["len"]
                len2 = origin_data[index2][file2]["len"]
                tree_seq1 = origin_data[index1][file1]["tree_seq"]
                tree_seq2 = origin_data[index2][file2]["tree_seq"]
                f.write(f"pred sim: {origin_pred[i]}\n")
                f.write(f"\n==========================================\n")
                f.write(f"origin_1 file: {file1n}\n")
                f.write(code1)
                f.write(f"\n-------------------------------------------\n")
                f.write(f"origin_2 file: {file1n}\n")
                f.write(code2)
            wrong_list_detail.append({"File1": file1n, "File2": file2n, "Code1": code1, "Code2": code2, "Tree1": tree_seq1, "Tree2": tree_seq2, "Len1": len1, "Len2": len2, "Result": labels[i], "Pred": origin_pred[i]})
        gold[labels[i]] += 1
    keys = list(wfdict.keys())
    values = list(wfdict.values())
    wfile = open(f'wrong/wronglist.txt', 'w')
    wfile.write("\n".join(wrong_names))
    with open(f'wrong/preds.csv', 'w', newline='') as csvfile:
        for i in range(len(all_names)):
            csvfile.write(all_names[i] + "," + str(all_pred[i]) + "," + str(all_gd[i]) + "," + str(all_right[i]) + "\n")
    keys = wrong_list_detail[0].keys()
    with open('wrong/wrong_list_detail.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(wrong_list_detail)
    print("  Prediction:", pred_all, " Right:", pred_right, " Gold:", gold)
    ''' -- for all labels -- '''
    print("  ****** Neg|Neu|Pos ******")
    accuracy = 1.0 * sum(pred_right) / total
    p, r, f1 = cal_prf(pred_all, pred_right, gold, formation=False)
    _, _, macro_f1 = cal_prf(pred_all, pred_right, gold,
                                    formation=False,
                                    metric_type="macro")
    print("    Accuracy on test is %d/%d = %f" % (sum(pred_right), total, accuracy))
    print("    Precision: %s\n    Recall   : %s\n    F1 score : %s\n " \
          % (p[1], r[1], f1[1]))

    if to_save:
        csv_path = "model/result.csv"
        if not os.path.exists(csv_path):
            rfile = open(csv_path, 'w')
            rfile.write('model,data,train_pair,test_pair,valid_pair,use_oast,output_dim,lr,epochs,batch_size,dropout,seed,transformer_nlayers,d_model,d_ff,attention_h,'
                        'p_1,r_1,f1_1,accuracy\n')
        else:
            rfile = open(csv_path, 'a')
        rfile.write(
            f'{args.model},{args.data},{args.train_pair},{args.test_pair},{args.valid_pair},{args.ast_type},{args.output_dim},{args.lr},{args.epochs},{args.batch_size},')
        rfile.write(
            f'{args.dropout},{args.seed},{args.transformer_nlayers},{args.d_model},{args.d_ff},{args.h},')
        rfile.write(
            f'{p[1]},{r[1]},{f1[1]},{accuracy}\n')

    return accuracy, f1[1]

def test_prf_noprint(pred, labels):
    total = len(labels)
    pred_right = [0, 0]
    pred_all = [0, 0]
    gold = [0, 0]
    for i in range(total):
        pred_all[pred[i]] += 1
        if pred[i] == labels[i]:
            pred_right[pred[i]] += 1
        gold[labels[i]] += 1

    accuracy = 1.0 * sum(pred_right) / total
    p, r, f1 = cal_prf(pred_all, pred_right, gold, formation=False)
    _, _, macro_f1 = cal_prf(pred_all, pred_right, gold,
                                    formation=False,
                                    metric_type="macro")
    return accuracy, macro_f1, p, r, f1


def test(model, dataset, args, criterion):
    test_set = dataset
    batches_per_step = int(len(test_set) / args.batch_size)
    tic = time.time()
    model.eval()
    pred = numpy.array([])
    labels = numpy.array([])
    loss = 0
    for step in tqdm(range(batches_per_step), position=0):
        with torch.no_grad():
            max_len, ast_token, sentences_seqlen, sentences_mask, label, others = test_set.next_batch(args.batch_size)
            assert len(ast_token[0]) == len(label)
            batch_size = len(ast_token[0])
            ast_token_, sentences_seqlen_, sentences_mask_= \
                var_batch(max_len, batch_size, ast_token, sentences_seqlen, sentences_mask, others)
            probs = model(ast_token_, sentences_seqlen_, sentences_mask_)          
            label_ = Variable(torch.FloatTensor(label))
            if args.cuda:
                label_ = label_.cuda()
            loss += criterion(probs, label_).item()

            if args.cuda:
                pred = numpy.concatenate((pred, probs.cpu().data.numpy()), axis=0)
            else:
                pred = numpy.concatenate((pred, probs.data.numpy()), axis=0)

            labels = numpy.concatenate((labels, label), axis=0)
            #print(pred)

    loss = loss / batches_per_step
    tit = time.time() - tic
    print("  Predicting {:d} examples using {:5.4f} seconds".format(len(test_set), tit))
    #labels = numpy.asarray(labels)

    binarypred = numpy.where(pred > args.valid_threshold, 1, 0)
    accuracy, f1 = test_prf(binarypred, labels.astype(int))
    return accuracy, f1, loss

def test_detail(model, dataset, args, to_save):
    test_set = dataset
    batches_per_step = int(len(test_set) / args.batch_size)
    tic = time.time()
    model.eval()
    pred = numpy.array([])
    labels = numpy.array([])
    for step in tqdm(range(batches_per_step), position=0):
        max_len, ast_token, sentences_seqlen, sentences_mask, label, others = test_set.next_batch(args.batch_size)
        assert len(ast_token[0]) == len(label)
        ''' Prepare data and prediction'''
        batch_size = len(ast_token[0])
        ast_token_, sentences_seqlen_, sentences_mask_= \
            var_batch(max_len, batch_size, ast_token, sentences_seqlen, sentences_mask, others)
        
        with torch.no_grad():
            probs = model(ast_token_, sentences_seqlen_, sentences_mask_)
        
        if args.cuda:
            pred = numpy.concatenate((pred, probs.cpu().data.numpy()), axis=0)
        else:
            pred = numpy.concatenate((pred, probs.data.numpy()), axis=0)

        labels = numpy.concatenate((labels, label), axis=0)

    tit = time.time() - tic
    print("  Predicting {:d} examples using {:5.4f} seconds".format(len(test_set), tit))

    #绘制P-R图和各种图
    d_threshold = numpy.arange(0.1, 1.0, 0.01)
    d_P = []
    d_R = []
    d_accuracy = []
    d_f1 = []
    best_F1 = 0
    best_th = 0
    for th in d_threshold:
        dpred = numpy.where(pred > th, 1, 0)
        accuracy, marco_f1, precision, recall, f1 = test_prf_noprint(dpred, labels.astype(int))
        d_P.append(precision[1])
        d_R.append(recall[1])
        d_accuracy.append(accuracy)
        if f1[1] > best_F1:
            best_F1 = f1[1]
            best_th = th
        d_f1.append(f1[1])
    plt.plot(d_P, d_R)
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.title('P-R')
    plt.savefig('pic/P_R.png')

    plt.clf()
    plt.plot(d_threshold, d_P)
    plt.xlabel('threshold')
    plt.ylabel('precision')
    plt.title('threshold-P')
    plt.savefig('pic/threshold_P.png')

    plt.clf()
    plt.plot(d_threshold, d_R)
    plt.xlabel('threshold')
    plt.ylabel('recall')
    plt.title('threshold-R')
    plt.savefig('pic/threshold_R.png')

    plt.clf()
    plt.plot(d_threshold, d_f1)
    plt.xlabel('threshold')
    plt.ylabel('f1')
    plt.title('threshold-f1')
    plt.savefig('pic/threshold_f1.png')

    plt.clf()
    plt.plot(d_threshold, d_accuracy)
    plt.xlabel('threshold')
    plt.ylabel('accuracy')
    plt.title('threshold-acc')
    plt.savefig('pic/threshold_acc.png')

    if args.threshold != 0:
        best_th = args.threshold
        print("---------> Defined Threshold: ", best_th)
    else:
        print("---------> Highest F1 Threshold: ", best_th)
    binarypred = numpy.where(pred > best_th, 1, 0)
    accuracy, f1 = test_prf_detail(binarypred, labels.astype(int), pred, test_set.clone_pair, test_set.processed_data
                                         , args, max_wrong=1000, to_save=to_save)
    return accuracy, f1


def var_batch(max_len, batch_size, ast_token, sentences_seqlen, sentences_mask, others):
    # dtype = torch.from_numpy(sentences, dtype=torch.cuda.LongTensor)
    ast_token_ = []
    sentences_mask_ = []
    sentences_seqlen_ = []
    ast_token_.append(Variable(torch.LongTensor(ast_token[0]).view(batch_size, max_len[0])))
    ast_token_.append(Variable(torch.LongTensor(ast_token[1]).view(batch_size, max_len[0])))
    ast_token_.append(Variable(torch.LongTensor(ast_token[2]).view(batch_size, max_len[1])))
    ast_token_.append(Variable(torch.LongTensor(ast_token[3]).view(batch_size, max_len[1])))
    sentences_seqlen_.append(Variable(torch.LongTensor(sentences_seqlen[0]).view(batch_size)))
    sentences_seqlen_.append(Variable(torch.LongTensor(sentences_seqlen[1]).view(batch_size)))
    sentences_mask_.append(Variable(torch.LongTensor(sentences_mask[0]).view(batch_size, max_len[0])))
    sentences_mask_.append(Variable(torch.LongTensor(sentences_mask[1]).view(batch_size, max_len[1])))

    ast_token_[0] = ast_token_[0].cuda()
    ast_token_[1] = ast_token_[1].cuda()
    ast_token_[2] = ast_token_[2].cuda()
    ast_token_[3] = ast_token_[3].cuda()
    sentences_seqlen_[0] = sentences_seqlen_[0].cuda()
    sentences_seqlen_[1] = sentences_seqlen_[1].cuda()
    sentences_mask_[0] = sentences_mask_[0].cuda()
    sentences_mask_[1] = sentences_mask_[1].cuda()

    return ast_token_, sentences_seqlen_, sentences_mask_

def train(model, training_data, args, optimizer, criterion):
    model.train()

    batch_size = args.batch_size

    max_len, ast_token, sentences_seqlen, sentences_mask, labels, others = training_data

    assert batch_size == len(ast_token[0]) == len(labels)
    ''' Prepare data and prediction'''
    ast_token_, sentences_seqlen_, sentences_mask_ = \
        var_batch(max_len, batch_size, ast_token, sentences_seqlen, sentences_mask, others)
    #print(ast_token_, sentences_mask_)
    labels_ = Variable(torch.FloatTensor(labels))
    if args.cuda:
        labels_ = labels_.cuda()

    assert len(ast_token_[0]) == len(labels)

    model.zero_grad()
    probs = model(ast_token_, sentences_seqlen_, sentences_mask_)
    loss = criterion(probs, labels_)

    loss.backward()
    optimizer.step()
    return loss.item()

def append_log_train_valid(loss, acc, f1, valid_loss, epoch, writer):
    writer.add_scalar('1_train/loss', loss, epoch)
    writer.add_scalar('2_valid/accuracy', acc, epoch)
    writer.add_scalar('2_valid/f1_score', f1, epoch)
    writer.add_scalar('2_valid/loss', valid_loss, epoch)

def append_log_test(acc, f1, epoch, writer):
    writer.add_scalar('3_test/accuracy', acc, epoch)
    writer.add_scalar('3_test/f1_score', f1, epoch)
