from torchtext.data import Example, Field

from rnng.actions import NTAction, GenAction, ReduceAction, ShiftAction
from rnng.example import make_example
from rnng.oracle import DiscOracle, GenOracle


def test_make_example_from_disc_oracle():
    actions = [
        NTAction('S'),
        NTAction('NP'),
        ShiftAction(),
        ReduceAction(),
        NTAction('VP'),
        ShiftAction(),
        NTAction('NP'),
        ShiftAction(),
        ReduceAction(),
        ReduceAction(),
        ReduceAction(),
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
    assert example.nonterms == [a.label for a in actions if isinstance(a, NTAction)]
    assert example.pos_tags == pos_tags
    assert example.words == words


def test_make_example_from_gen_oracle():
    actions = [
        NTAction('S'),
        NTAction('NP'),
        GenAction('John'),
        ReduceAction(),
        NTAction('VP'),
        GenAction('loves'),
        NTAction('NP'),
        GenAction('Mary'),
        ReduceAction(),
        ReduceAction(),
        ReduceAction(),
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
    assert example.nonterms == [a.label for a in actions if isinstance(a, NTAction)]
    assert example.pos_tags == pos_tags
    assert example.words == words
