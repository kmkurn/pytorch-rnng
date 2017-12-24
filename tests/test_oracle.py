from nltk.tree import Tree
import pytest

from rnng.actions import ShiftAction, ReduceAction, NTAction, GenAction
from rnng.oracle import DiscOracle, GenOracle


class TestDiscOracle:
    def test_init(self):
        actions = [NTAction('S'), ShiftAction()]
        pos_tags = ['NNP']
        words = ['John']

        oracle = DiscOracle(actions, pos_tags, words)

        assert oracle.actions == actions
        assert oracle.pos_tags == pos_tags
        assert oracle.words == words

    def test_init_with_unequal_shift_count_and_number_of_words(self):
        actions = [NTAction('S')]
        pos_tags = ['NNP']
        words = ['John']
        with pytest.raises(ValueError) as excinfo:
            DiscOracle(actions, pos_tags, words)
        assert 'number of words should match number of SHIFT actions' in str(excinfo.value)

    def test_init_with_unequal_number_of_words_and_pos_tags(self):
        actions = [NTAction('S'), ShiftAction()]
        pos_tags = ['NNP', 'VBZ']
        words = ['John']
        with pytest.raises(ValueError) as excinfo:
            DiscOracle(actions, pos_tags, words)
        assert 'number of POS tags should match number of words' in str(excinfo.value)

    def test_from_parsed_sent(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
        expected_actions = [
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
        expected_pos_tags = ['NNP', 'VBZ', 'NNP']
        expected_words = ['John', 'loves', 'Mary']

        oracle = DiscOracle.from_parsed_sent(Tree.fromstring(s))

        assert isinstance(oracle, DiscOracle)
        assert oracle.actions == expected_actions
        assert oracle.pos_tags == expected_pos_tags
        assert oracle.words == expected_words


class TestGenOracle:
    def test_init_with_unequal_gen_count_and_number_of_pos_tags(self):
        actions = [NTAction('S')]
        pos_tags = ['NNP']
        with pytest.raises(ValueError) as excinfo:
            GenOracle(actions, pos_tags)
        assert 'number of POS tags should match number of GEN actions' in str(excinfo.value)

    def test_from_parsed_sent(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
        expected_actions = [
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
            ReduceAction()
        ]
        expected_words = ['John', 'loves', 'Mary']
        expected_pos_tags = ['NNP', 'VBZ', 'NNP']

        oracle = GenOracle.from_parsed_sent(Tree.fromstring(s))

        assert isinstance(oracle, GenOracle)
        assert oracle.actions == expected_actions
        assert oracle.words == expected_words
        assert oracle.pos_tags == expected_pos_tags
