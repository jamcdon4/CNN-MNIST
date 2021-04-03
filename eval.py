#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: eval.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import csv
import numpy as np

RESULT_PATH = os.getcwd()
GT_LIST = [1, 9, 2, 3]

def read_result(result_path, file_name):
    file_path = os.path.join(result_path, file_name)
    pred_list = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            pred = list(map(int, line.split(',')))
            pred_list.append(np.argmax(pred))
    return pred_list


def eval_accuracy(gt, pred):
    gt = np.array(gt)
    pred = np.array(pred)
    return np.mean((np.equal(gt, pred).astype(int)))

def main():

    pred = read_result(RESULT_PATH, 'mnist.csv')
    n_test_sample = len(pred)
    gt_list = np.random.choice(10, n_test_sample)
    accuracy = eval_accuracy(gt_list, pred)
    print('[MNIST Classification] number of test: {}, accuracy: {}'
          .format(n_test_sample, accuracy))

if __name__ == "__main__":
    main()