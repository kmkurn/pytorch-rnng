import pytest

from rnng.actions import Action, ShiftAction, ReduceAction, NTAction, GenAction


def test_action_from_invalid_string():
    with pytest.raises(ValueError) as excinfo:
        Action.from_string('asdf')
    assert f'no action found from string asdf' in str(excinfo.value)


class TestShiftAction(object):
    as_str = 'SHIFT'

    @staticmethod
    def make_action():
        return ShiftAction()

    def test_eq(self):
        assert self.make_action() == self.make_action()
        assert self.make_action() != 'foo'

    def test_hash(self):
        assert hash(self.make_action()) == hash(self.make_action())

    def test_str(self):
        assert str(self.make_action()) == self.as_str

    def test_from_string(self):
        a = Action.from_string(self.as_str)
        assert isinstance(a, ShiftAction)


class TestReduceAction(object):
    as_str = 'REDUCE'

    @staticmethod
    def make_action():
        return ReduceAction()

    def test_eq(self):
        assert self.make_action() == self.make_action()
        assert self.make_action() != 'foo'

    def test_hash(self):
        assert hash(self.make_action()) == hash(self.make_action())

    def test_str(self):
        assert str(self.make_action()) == self.as_str

    def test_from_string(self):
        a = Action.from_string(self.as_str)
        assert isinstance(a, ReduceAction)


class TestNTAction(object):
    as_str = 'NT({label})'

    @staticmethod
    def make_action(label='NP'):
        return NTAction(label)

    def test_eq(self):
        assert self.make_action() == self.make_action()
        assert self.make_action() != self.make_action('VP')
        assert self.make_action() != 'foo'

    def test_hash(self):
        assert hash(self.make_action()) == hash(self.make_action())
        assert hash(self.make_action()) != hash(self.make_action('VP'))

    def test_str(self):
        a = self.make_action()
        assert str(a) == self.as_str.format(label=a.label)

    def test_from_string(self):
        label = 'NP'
        a = Action.from_string(self.as_str.format(label=label))
        assert isinstance(a, NTAction)
        assert a.label == label


class TestGenAction(object):
    as_str = 'GEN({word})'

    @staticmethod
    def make_action(word='asdf'):
        return GenAction(word)

    def test_eq(self):
        assert self.make_action() == self.make_action()
        assert self.make_action() != self.make_action('fdsa')
        assert self.make_action() != 'foo'

    def test_hash(self):
        assert hash(self.make_action()) == hash(self.make_action())
        assert hash(self.make_action()) != hash(self.make_action('fdsa'))

    def test_str(self):
        a = self.make_action()
        assert str(a) == self.as_str.format(word=a.word)

    def test_from_string(self):
        word = 'asdf'
        a = Action.from_string(self.as_str.format(word=word))
        assert isinstance(a, GenAction)
        assert a.word == word
