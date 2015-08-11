#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import csv

import logging

class SpliceJunction(object):

    def __init__(self, path='data/splice_junction/splice-junction-gene-sequences/splice.data'):
        dataset = []

        with open(path, 'rb') as csvf:
            rows = csv.reader(csvf, delimiter=',')
            for row in rows:
                dataset += [[e.strip() for e in row]]

        symbols = {
            'A': [1, 0, 0, 0, 0, 0, 0, 0],
            'G': [0, 1, 0, 0, 0, 0, 0, 0],
            'T': [0, 0, 1, 0, 0, 0, 0, 0],
            'C': [0, 0, 0, 1, 0, 0, 0, 0],

            'D': [0, 0, 0, 0, 1, 0, 0, 0],
            'N': [0, 0, 0, 0, 0, 1, 0, 0],
            'S': [0, 0, 0, 0, 0, 0, 1, 0],
            'R': [0, 0, 0, 0, 0, 0, 0, 1]
        }

        classes = {
            'EI': [0], 'IE': [1], 'N': [2]
        }

        self.labels = [classes[i[0]] for i in dataset]
        self.sequences = [[symbols[j] for j in i[2]] for i in dataset]

if __name__ == '__main__':
    p = Promoter()
