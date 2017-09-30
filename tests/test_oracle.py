from nltk.tree import Tree
import pytest

from rnng.oracle import ShiftAction, ReduceAction, NTAction, GenAction, DiscOracle, GenOracle


class TestShiftAction:
    def test_to_string(self):
        a = ShiftAction()
        assert str(a) == 'SHIFT'

    def test_from_string(self):
        a = ShiftAction.from_string('SHIFT')
        assert isinstance(a, ShiftAction)

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            ShiftAction.from_string('asdf')


class TestReduceAction:
    def test_to_string(self):
        a = ReduceAction()
        assert str(a) == 'REDUCE'

    def test_from_string(self):
        a = ReduceAction.from_string('REDUCE')
        assert isinstance(a, ReduceAction)

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            ShiftAction.from_string('asdf')


class TestNTAction:
    def test_to_string(self):
        a = NTAction('NP')
        assert str(a) == 'NT(NP)'

    def test_from_string(self):
        a = NTAction.from_string('NT(NP)')
        assert isinstance(a, NTAction)
        assert a.label == 'NP'

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            NTAction.from_string('asdf')


class TestGenAction:
    def test_to_string(self):
        a = GenAction('asdf')
        assert str(a) == 'GEN(asdf)'

    def test_from_string(self):
        a = GenAction.from_string('GEN(asdf)')
        assert isinstance(a, GenAction)
        assert a.word == 'asdf'

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            GenAction.from_string('asdf')


class TestDiscOracle:
    def test_from_parsed_sent(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
        expected_actions = ['NT(S)', 'NT(NP)', 'SHIFT', 'REDUCE', 'NT(VP)', 'SHIFT', 'NT(NP)',
                            'SHIFT', 'REDUCE', 'REDUCE', 'REDUCE']
        expected_pos_tags = ['NNP', 'VBZ', 'NNP']
        expected_words = ['John', 'loves', 'Mary']

        oracle = DiscOracle.from_parsed_sent(Tree.fromstring(s))

        assert isinstance(oracle, DiscOracle)
        assert [str(a) for a in oracle.actions] == expected_actions
        assert oracle.pos_tags == expected_pos_tags
        assert oracle.words == expected_words

    def test_from_string(self):
        s = 'asdf fdsa\nNNP VBZ\nNT(S)\nSHIFT\nSHIFT\nREDUCE'

        oracle = DiscOracle.from_string(s)

        assert isinstance(oracle, DiscOracle)
        assert oracle.words == ['asdf', 'fdsa']
        assert oracle.pos_tags == ['NNP', 'VBZ']
        assert [str(a) for a in oracle.actions] == ['NT(S)', 'SHIFT', 'SHIFT', 'REDUCE']

    def test_from_string_too_short(self):
        s = 'asdf asdf\nNT(S)\nSHIFT\nSHIFT\nREDUCE'

        with pytest.raises(ValueError):
            DiscOracle.from_string(s)


class TestGenOracle:
    def test_from_parsed_sent(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
        expected_actions = ['NT(S)', 'NT(NP)', 'GEN(John)', 'REDUCE', 'NT(VP)', 'GEN(loves)',
                            'NT(NP)', 'GEN(Mary)', 'REDUCE', 'REDUCE', 'REDUCE']
        expected_words = ['John', 'loves', 'Mary']
        expected_pos_tags = ['NNP', 'VBZ', 'NNP']

        oracle = GenOracle.from_parsed_sent(Tree.fromstring(s))

        assert isinstance(oracle, GenOracle)
        assert [str(a) for a in oracle.actions] == expected_actions
        assert oracle.words == expected_words
        assert oracle.pos_tags == expected_pos_tags

    def test_from_string(self):
        s = 'NNP VBZ\nNT(S)\nGEN(asdf)\nGEN(fdsa)\nREDUCE'

        oracle = GenOracle.from_string(s)

        assert isinstance(oracle, GenOracle)
        assert oracle.words == ['asdf', 'fdsa']
        assert oracle.pos_tags == ['NNP', 'VBZ']
        assert [str(a) for a in oracle.actions] == ['NT(S)', 'GEN(asdf)', 'GEN(fdsa)',
                                                    'REDUCE']

    def test_from_string_too_short(self):
        s = 'NT(S)'

        with pytest.raises(ValueError):
            GenOracle.from_string(s)
