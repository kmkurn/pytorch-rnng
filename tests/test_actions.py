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

    def test_execute_on(self, mocker):
        a = self.make_action()
        fake_parser = mocker.Mock()
        a.execute_on(fake_parser)
        fake_parser.shift.assert_called_once_with()

    def test_from_string(self):
        a = Action.from_string(self.as_str)
        assert isinstance(a, ShiftAction)

    def test_verify_on(self, mocker):
        a = self.make_action()
        fake_parser = mocker.Mock()
        a.verify_on(fake_parser)
        fake_parser.verify_shift.assert_called_once_with()


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

    def test_execute_on(self, mocker):
        a = self.make_action()
        fake_parser = mocker.Mock()
        a.execute_on(fake_parser)
        fake_parser.reduce.assert_called_once_with()

    def test_from_string(self):
        a = Action.from_string(self.as_str)
        assert isinstance(a, ReduceAction)

    def test_verify_on(self, mocker):
        a = self.make_action()
        fake_parser = mocker.Mock()
        a.verify_on(fake_parser)
        fake_parser.verify_reduce.assert_called_once_with()


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

    def test_execute_on(self, mocker):
        a = self.make_action()
        fake_parser = mocker.Mock()
        a.execute_on(fake_parser)
        fake_parser.push_nt.assert_called_once_with(a.label)

    def test_from_string(self):
        label = 'NP'
        a = Action.from_string(self.as_str.format(label=label))
        assert isinstance(a, NTAction)
        assert a.label == label

    def test_verify_on(self, mocker):
        a = self.make_action()
        fake_parser = mocker.Mock()
        a.verify_on(fake_parser)
        fake_parser.verify_push_nt.assert_called_once_with()


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
