#!/usr/bin/env python

from argparse import ArgumentParser
import pickle

from rnng.oracle import DiscOracle, GenOracle
from rnng.vocab import Vocabulary


parser = ArgumentParser(description='Get vocabulary from an oracle file')
parser.add_argument('oracle', help='path to oracle file')
parser.add_argument('--outfile', '-o', required=True, help='where to save the vocabulary to')
parser.add_argument('--generative', '-g', action='store_true', default=False,
                    help='whether the oracle file is for the generative parser')
parser.add_argument('--min-count', '-c', type=int, default=2,
                    help='minimum word count to be included in the vocabulary (default: 2)')
args = parser.parse_args()

vocab = Vocabulary(min_count=args.min_count)
oracle_class = GenOracle if args.generative else DiscOracle
with open(args.oracle) as f:
    oracles = [oracle_class.from_string(oracle_str)
               for oracle_str in f.read().split('\n\n') if oracle_str]
vocab.load_oracles(oracles)
with open(args.outfile, 'wb') as f:
    pickle.dump(vocab, f)
