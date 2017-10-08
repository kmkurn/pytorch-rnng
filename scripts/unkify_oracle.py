#!/usr/bin/env python -O

from argparse import ArgumentParser
from collections import Counter

from rnng.actions import GenAction, NTAction
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
vocab_words = set()
vocab_pos = set()
vocab_nt = set()
for oracle in oracles:
    vocab_words.update([w for w in oracle.words if counter[w] >= args.min_count])
    vocab_pos.update(oracle.pos_tags)
    vocab_nt.update([a.label for a in oracle.actions if isinstance(a, NTAction)])

oracles = read_oracles_from_file(oracle_class, args.oracle_file)
for oracle in oracles:
    if set(oracle.pos_tags) - vocab_pos:
        continue
    nt_labels = {a.label for a in oracle.actions if isinstance(a, NTAction)}
    if nt_labels - vocab_nt:
        continue
    oracle.words = [w if w in vocab_words else 'UNK' for w in oracle.words]
    print(oracle, end='\n\n')
