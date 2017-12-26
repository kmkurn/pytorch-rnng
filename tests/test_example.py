from torchtext.data import Example, Field

from rnng.actions import NT, GEN, REDUCE, SHIFT, get_nonterm, is_nt
from rnng.example import make_example
from rnng.oracle import DiscOracle, GenOracle


def test_make_example_from_disc_oracle():
    actions = [
        NT('S'),
        NT('NP'),
        SHIFT,
        REDUCE,
        NT('VP'),
        SHIFT,
        NT('NP'),
        SHIFT,
        REDUCE,
        REDUCE,
        REDUCE,
    ]
    pos_tags = 'NNP VBZ NNP'.split()
    words = 'John loves Mary'.split()
    oracle = DiscOracle(actions, pos_tags, words)
    fields = [
        ('actions', Field()),
        ('nonterms', Field()),
        ('pos_tags', Field()),
        ('words', Field()),
    ]

    example = make_example(oracle, fields)

    assert isinstance(example, Example)
    assert example.actions == actions
    assert example.nonterms == [get_nonterm(a) for a in actions if is_nt(a)]
    assert example.pos_tags == pos_tags
    assert example.words == words


def test_make_example_from_gen_oracle():
    actions = [
        NT('S'),
        NT('NP'),
        GEN('John'),
        REDUCE,
        NT('VP'),
        GEN('loves'),
        NT('NP'),
        GEN('Mary'),
        REDUCE,
        REDUCE,
        REDUCE,
    ]
    pos_tags = 'NNP VBZ NNP'.split()
    words = 'John loves Mary'.split()
    oracle = GenOracle(actions, pos_tags)
    fields = [
        ('actions', Field()),
        ('nonterms', Field()),
        ('pos_tags', Field()),
        ('words', Field()),
    ]

    example = make_example(oracle, fields)

    assert isinstance(example, Example)
    assert example.actions == actions
    assert example.nonterms == [get_nonterm(a) for a in actions if is_nt(a)]
    assert example.pos_tags == pos_tags
    assert example.words == words
