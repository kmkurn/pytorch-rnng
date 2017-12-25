import argparse

import rnng.commands.train as train


def make_parser():
    parser = argparse.ArgumentParser(description='Command line interface to RNNG.')
    subparsers = parser.add_subparsers()
    train.make_parser(subparsers)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_usage()
