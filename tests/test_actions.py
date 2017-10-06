import pytest

from rnng.actions import ShiftAction, ReduceAction, NTAction, GenAction


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
            ReduceAction.from_string('asdf')


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
