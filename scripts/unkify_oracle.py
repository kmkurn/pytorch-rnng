#!/usr/bin/env python -O

from argparse import ArgumentParser
from collections import Counter

from rnng.oracle import DiscOracle, GenOracle


def read_oracles_from_file(oracle_class, filename):
    with open(filename) as f:
        oracles = [oracle_class.from_string(oracle_str)
                   for oracle_str in f.read().split('\n\n') if oracle_str]
    return oracles


parser = ArgumentParser(description='Unkify an oracle file based on the given training oracle')
parser.add_argument('training_file', help='path to training oracle file')
parser.add_argument('oracle_file', help='path to oracle file to unkify')
parser.add_argument('--generative', '-g', action='store_true', default=False,
                    help='whether the oracle files are for the generative parser')
parser.add_argument('--min-count', '-c', type=int, default=2,
                    help='minimum word count to be included in the vocabulary (default: 2)')
args = parser.parse_args()

oracle_class = GenOracle if args.generative else DiscOracle
oracles = read_oracles_from_file(oracle_class, args.training_file)

counter = Counter([w for oracle in oracles for w in oracle.words])

oracles = read_oracles_from_file(oracle_class, args.oracle_file)
for oracle in oracles:
    oracle.words = [w if counter[w] >= args.min_count else 'UNK' for w in oracle.words]
    print(oracle, end='\n\n')
