import pytest

from rnng.actions import ShiftAction, ReduceAction, NTAction, GenAction


class TestShiftAction:
    as_str = 'SHIFT'

    def test_to_string(self):
        a = ShiftAction()
        assert str(a) == self.as_str

    def test_hash(self):
        a = ShiftAction()
        assert hash(a) == hash(self.as_str)

    def test_eq(self):
        assert ShiftAction() == ShiftAction()
        assert ShiftAction() != ReduceAction()

    def test_from_string(self):
        a = ShiftAction.from_string(self.as_str)
        assert isinstance(a, ShiftAction)

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            ShiftAction.from_string('asdf')


class TestReduceAction:
    as_str = 'REDUCE'

    def test_to_string(self):
        a = ReduceAction()
        assert str(a) == self.as_str

    def test_hash(self):
        a = ReduceAction()
        assert hash(a) == hash(self.as_str)

    def test_eq(self):
        assert ReduceAction() == ReduceAction()
        assert ReduceAction() != ShiftAction()

    def test_from_string(self):
        a = ReduceAction.from_string(self.as_str)
        assert isinstance(a, ReduceAction)

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            ReduceAction.from_string('asdf')


class TestNTAction:
    as_str = 'NT({label})'

    def test_to_string(self):
        label = 'NP'
        a = NTAction(label)
        assert str(a) == self.as_str.format(label=label)

    def test_hash(self):
        label = 'NP'
        a = NTAction(label)
        assert hash(a) == hash(self.as_str.format(label=label))

    def test_eq(self):
        a = NTAction('NP')
        assert a == NTAction(a.label)
        assert a != NTAction('asdf')
        assert a != ShiftAction()

    def test_from_string(self):
        label = 'NP'
        a = NTAction.from_string(self.as_str.format(label=label))
        assert isinstance(a, NTAction)
        assert a.label == label

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            NTAction.from_string('asdf')


class TestGenAction:
    as_str = 'GEN({word})'

    def test_to_string(self):
        word = 'asdf'
        a = GenAction(word)
        assert str(a) == self.as_str.format(word=word)

    def test_hash(self):
        word = 'asdf'
        a = GenAction(word)
        assert hash(a) == hash(self.as_str.format(word=word))

    def test_eq(self):
        a = GenAction('asdf')
        assert a == GenAction(a.word)
        assert a != GenAction('fdsa')
        assert a != ReduceAction()

    def test_from_string(self):
        word = 'asdf'
        a = GenAction.from_string(self.as_str.format(word=word))
        assert isinstance(a, GenAction)
        assert a.word == word

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            GenAction.from_string('asdf')
