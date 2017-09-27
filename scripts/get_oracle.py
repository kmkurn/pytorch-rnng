#!/usr/bin/env python

from argparse import ArgumentParser
import sys

from rnng.oracle import DiscOracle, GenOracle, GenAction


parser = ArgumentParser(description='Get oracle for a given corpus')
parser.add_argument('corpus', help="corpus file, or 'STDIN' to read from standard input")
parser.add_argument('--vocab', required=True, help='vocabulary file')
parser.add_argument('--generative', '-g', action='store_true', default=False,
                    help='get oracle for the generative parser')
parser.add_argument('--no-lowercase', action='store_false', default=True, dest='lowercase',
                    help='do not lowercase words')
args = parser.parse_args()

with open(args.vocab) as f:
    vocab = {line.strip() for line in f}

try:
    if args.corpus == 'STDIN':
        infile = sys.stdin
    else:
        infile = open(args.corpus)

    for line in infile:
        if args.generative:
            oracle = GenOracle.from_bracketed_string(line.strip())
            if args.lowercase:
                oracle.actions = [GenAction(a.word.lower()) if isinstance(a, GenAction) else a
                                  for a in oracle.actions]
            oracle.actions = [GenAction('UNK') if isinstance(a, GenAction) and
                              a.word not in vocab else a for a in oracle.actions]
        else:
            oracle = DiscOracle.from_bracketed_string(line.strip())
            if args.lowercase:
                oracle.words = [w.lower() for w in oracle.words]
            oracle.words = ['UNK' if w not in vocab else w for w in oracle.words]
        print(str(oracle), end='\n\n')
finally:
    if infile is not sys.stdin:
        infile.close()
