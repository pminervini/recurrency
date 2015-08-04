#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import csv

import logging

class Promoter:

    def __init__(self, path='data/promoter/promoter-gene-sequences/promoters.data'):
        dataset = []

        with open(path, 'rb') as csvf:
            rows = csv.reader(csvf, delimiter=',')
            for row in rows:
                dataset += [[e.strip() for e in row]]

        symbols = {
            'a': [1, 0, 0, 0],
            'g': [0, 1, 0, 0],
            't': [0, 0, 1, 0],
            'c': [0, 0, 0, 1],
            '-': [0], '+': [1]
        }

        self.labels = [symbols[i[0]] for i in dataset]
        self.sequences = [[symbols[j] for j in i[2]] for i in dataset]
