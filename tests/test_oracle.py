from nltk.tree import Tree
import pytest

from rnng.actions import ShiftAction, ReduceAction, NTAction, GenAction
from rnng.oracle import DiscOracle, GenOracle, OracleDataset
from rnng.utils import TermStore


class TestDiscOracle:
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

    def test_from_string(self):
        s = 'asdf fdsa\nNNP VBZ\nNT(S)\nSHIFT\nSHIFT\nREDUCE'

        oracle = DiscOracle.from_string(s)

        assert isinstance(oracle, DiscOracle)
        assert oracle.words == ['asdf', 'fdsa']
        assert oracle.pos_tags == ['NNP', 'VBZ']
        assert oracle.actions == [NTAction('S'), ShiftAction(), ShiftAction(), ReduceAction()]

    def test_from_string_too_short(self):
        s = 'asdf asdf\nNT(S)\nSHIFT\nSHIFT\nREDUCE'

        with pytest.raises(ValueError):
            DiscOracle.from_string(s)


class TestGenOracle:
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

    def test_from_string(self):
        s = 'NNP VBZ\nNT(S)\nGEN(asdf)\nGEN(fdsa)\nREDUCE'

        oracle = GenOracle.from_string(s)

        assert isinstance(oracle, GenOracle)
        assert oracle.words == ['asdf', 'fdsa']
        assert oracle.pos_tags == ['NNP', 'VBZ']
        assert oracle.actions == [NTAction('S'), GenAction('asdf'), GenAction('fdsa'),
                                  ReduceAction()]

    def test_from_string_too_short(self):
        s = 'NT(S)'

        with pytest.raises(ValueError):
            GenOracle.from_string(s)


class TestOracleDataset:
    bracketed_sents = [
        '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))',
        '(S (NP (NNP Mary)) (VP (VBZ hates) (NP (NNP John))))'  # poor John
    ]
    words = {'John', 'loves', 'hates', 'Mary'}
    pos_tags = {'NNP', 'VBZ'}
    nt_labels = {'S', 'NP', 'VP'}
    actions = {str(NTAction('S')), str(NTAction('NP')), str(NTAction('VP')),
               str(ShiftAction()), str(ReduceAction())}

    def test_init(self):
        oracles = [DiscOracle.from_parsed_sent(Tree.fromstring(s))
                   for s in self.bracketed_sents]

        dataset = OracleDataset(oracles)

        assert isinstance(dataset.word_store, TermStore)
        assert set(dataset.word_store) == self.words
        assert isinstance(dataset.pos_store, TermStore)
        assert set(dataset.pos_store) == self.pos_tags
        assert isinstance(dataset.nt_store, TermStore)
        assert set(dataset.nt_store) == self.nt_labels
        assert isinstance(dataset.action_store, TermStore)
        assert set(dataset.action_store) == self.actions
        assert len(dataset.nt2action) == len(self.nt_labels)
        for label in self.nt_labels:
            nt_id = dataset.nt_store.get_id(label)
            action_id = dataset.nt2action[nt_id]
            assert action_id == dataset.action_store.get_id(str(NTAction(label)))

    def test_getitem(self):
        oracles = [DiscOracle.from_parsed_sent(Tree.fromstring(s))
                   for s in self.bracketed_sents]
        dataset = OracleDataset(oracles)

        assert oracles[0] is dataset[0]
        assert oracles[1] is dataset[1]
