import argparse
import os
import numpy
import math
import torch
import time
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.autograd import Variable
from tutils import test_prf, test_prf_detail
from concurrent.futures import ThreadPoolExecutor

def quick_test(model, dataset, args, to_save=False, threshold=0.5):
    print("Begin to generate code features")
    tic = time.time()

    index, file, pre, post, seq_len, mask_matrix, max_len = dataset.all_file_data()
    batch_size = args.batch_size
    gen_steps = int(len(file) / batch_size) + 1
    all_features = {}
    for i in tqdm(range(gen_steps)):
        start = i * batch_size
        end = start + batch_size
        if end > len(file):
            end = len(file)
        batch_index = index[start: end]
        batch_file = file[start: end]
        batch_pre = pre[start: end]
        batch_post = post[start: end]
        batch_seq_len = seq_len[start: end]
        batch_mask_matrix = mask_matrix[start: end]

        current_batch_size = len(batch_pre)
        batch_pre = Variable(torch.LongTensor(batch_pre).view(current_batch_size, max_len)).cuda()
        batch_post = Variable(torch.LongTensor(batch_post).view(current_batch_size, max_len)).cuda()
        batch_seq_len = Variable(torch.LongTensor(batch_seq_len).view(current_batch_size)).cuda()
        batch_mask_matrix = Variable(torch.LongTensor(batch_mask_matrix).view(current_batch_size, max_len)).cuda()

        model.eval()
        with torch.no_grad():
            features = model.feature_gen(batch_pre, batch_post, batch_seq_len, batch_mask_matrix)

        for i in range(current_batch_size):
            code_index = batch_index[i]
            code_file = batch_file[i]
            code_feature = features[i]
            if code_index not in all_features:
                all_features[code_index] = {}
            all_features[code_index][code_file] = code_feature.unsqueeze(0)

    pair_datas = dataset.clone_pair
    print("\n[In Test]Generate code features using %.6f seconds" % (time.time() - tic))

    print("Begin to calculate similarity")
    tic = time.time()
    feature1_list = []
    feature2_list = []
    batch_cos = args.batch_size * 4
    labels = []
    pred = numpy.array([])
    for i in tqdm(range(len(pair_datas))):
        pair = pair_datas[i]
        index1 = pair["index1"]
        index2 = pair["index2"]
        file1 = pair["file1"]
        file2 = pair["file2"]
        label = pair["label"]

        feature1 = all_features[index1][file1]
        feature2 = all_features[index2][file2]

        feature1_list.append(feature1)
        feature2_list.append(feature2)
        labels.append(label)
        if (i+1) % batch_cos == 0 or i == len(pair_datas)-1:
            feature1_tensor = torch.cat(feature1_list, dim=0)
            feature2_tensor = torch.cat(feature2_list, dim=0)
            sim = torch.clamp(F.cosine_similarity(feature1_tensor, feature2_tensor), min=0, max=1)
            pred = numpy.concatenate((pred, sim.cpu().data.numpy()), axis=0)
            feature1_list = []
            feature2_list = []

    labels = numpy.array(labels)
    binarypred = numpy.where(pred > threshold, 1, 0)
    #accuracy, f1 = test_prf_detail(binarypred, labels.astype(int), pred, dataset.clone_pair, dataset.processed_data
    #                                    , args, max_wrong=1000, to_save=to_save)
    accuracy, f1 = test_prf(binarypred, labels)
    print("\n[In Test]Calculate code similarity using %.6f seconds" % (time.time() - tic))
    return accuracy, f1, 0