from argparse import ArgumentParser
from collections import defaultdict
import sys

from rnng.oracle import DiscOracle


parser = ArgumentParser(
    description='Get vocabulary from a training corpus and print to stdout')
parser.add_argument('file',
                    help="training corpus file, or 'STDIN' to read from standard input")
parser.add_argument('--min-count', '-c', type=int, default=2,
                    help='minimum word count to be included in the vocabulary (default: 2)')
parser.add_argument('--no-lowercase', action='store_false', default=True, dest='lowercase',
                    help='do not lowercase words')
args = parser.parse_args()

try:
    if args.file == 'STDIN':
        infile = sys.stdin
    else:
        infile = open(args.file)

    count = defaultdict(int)
    for line in infile:
        oracle = DiscOracle.from_bracketed_string(line.strip())
        for w in oracle.words:
            if args.lowercase:
                w = w.lower()
            count[w] += 1
finally:
    if infile is not sys.stdin:
        infile.close()

for word, cnt in sorted(count.items()):
    if cnt >= args.min_count:
        print(word)
