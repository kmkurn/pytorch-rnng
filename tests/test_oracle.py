from nltk.tree import Tree
import pytest

from rnng.actions import GEN, NT, REDUCE, SHIFT
from rnng.oracle import DiscOracle, GenOracle


class TestDiscOracle:
    def test_init(self):
        actions = [NT('S'), SHIFT]
        pos_tags = ['NNP']
        words = ['John']

        oracle = DiscOracle(actions, pos_tags, words)

        assert oracle.actions == actions
        assert oracle.pos_tags == pos_tags
        assert oracle.words == words

    def test_init_with_unequal_shift_count_and_number_of_words(self):
        actions = [NT('S')]
        pos_tags = ['NNP']
        words = ['John']
        with pytest.raises(ValueError) as excinfo:
            DiscOracle(actions, pos_tags, words)
        assert 'number of words should match number of SHIFT actions' in str(excinfo.value)

    def test_init_with_unequal_number_of_words_and_pos_tags(self):
        actions = [NT('S'), SHIFT]
        pos_tags = ['NNP', 'VBZ']
        words = ['John']
        with pytest.raises(ValueError) as excinfo:
            DiscOracle(actions, pos_tags, words)
        assert 'number of POS tags should match number of words' in str(excinfo.value)

    def test_from_tree(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
        expected_actions = [
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
        expected_pos_tags = ['NNP', 'VBZ', 'NNP']
        expected_words = ['John', 'loves', 'Mary']

        oracle = DiscOracle.from_tree(Tree.fromstring(s))

        assert isinstance(oracle, DiscOracle)
        assert oracle.actions == expected_actions
        assert oracle.pos_tags == expected_pos_tags
        assert oracle.words == expected_words

    def test_to_tree(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
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
        pos_tags = ['NNP', 'VBZ', 'NNP']
        words = ['John', 'loves', 'Mary']

        oracle = DiscOracle(actions, pos_tags, words)

        assert str(oracle.to_tree()) == s


class TestGenOracle:
    def test_init_with_unequal_gen_count_and_number_of_pos_tags(self):
        actions = [NT('S')]
        pos_tags = ['NNP']
        with pytest.raises(ValueError) as excinfo:
            GenOracle(actions, pos_tags)
        assert 'number of POS tags should match number of GEN actions' in str(excinfo.value)

    def test_from_tree(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
        expected_actions = [
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
            REDUCE
        ]
        expected_words = ['John', 'loves', 'Mary']
        expected_pos_tags = ['NNP', 'VBZ', 'NNP']

        oracle = GenOracle.from_tree(Tree.fromstring(s))

        assert isinstance(oracle, GenOracle)
        assert oracle.actions == expected_actions
        assert oracle.words == expected_words
        assert oracle.pos_tags == expected_pos_tags

    def test_to_tree(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
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
        pos_tags = ['NNP', 'VBZ', 'NNP']

        oracle = GenOracle(actions, pos_tags)

        assert str(oracle.to_tree()) == s
