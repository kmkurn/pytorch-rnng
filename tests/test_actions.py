import pytest

from rnng.actions import ShiftAction, ReduceAction, NTAction, GenAction


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
        a = ShiftAction.from_string(self.as_str)
        assert isinstance(a, ShiftAction)

    def test_from_invalid_string(self):
        with pytest.raises(ValueError) as excinfo:
            ShiftAction.from_string('asdf')
        assert f'invalid string value for {self.as_str} action' in str(excinfo.value)

    def test_verify_on(self, mocker):
        a = self.make_action()
        fake_parser = mocker.Mock()
        a.verify_on(fake_parser)
        fake_parser.verify_shift.assert_called_once_with()


class TestReduceAction:
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
        a = ReduceAction.from_string(self.as_str)
        assert isinstance(a, ReduceAction)

    def test_from_invalid_string(self):
        with pytest.raises(ValueError) as excinfo:
            ReduceAction.from_string('asdf')
        assert f'invalid string value for {self.as_str} action' in str(excinfo.value)

    def test_verify_on(self, mocker):
        a = self.make_action()
        fake_parser = mocker.Mock()
        a.verify_on(fake_parser)
        fake_parser.verify_reduce.assert_called_once_with()


class TestNTAction:
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
        a = NTAction.from_string(self.as_str.format(label=label))
        assert isinstance(a, NTAction)
        assert a.label == label

    def test_from_invalid_string(self):
        with pytest.raises(ValueError) as excinfo:
            NTAction.from_string('asdf')
        as_str = self.as_str.format(label='X')
        assert f'invalid string value for {as_str} action' in str(excinfo.value)

    def test_verify_on(self, mocker):
        a = self.make_action()
        fake_parser = mocker.Mock()
        a.verify_on(fake_parser)
        fake_parser.verify_push_nt.assert_called_once_with()


class TestGenAction:
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
        a = GenAction.from_string(self.as_str.format(word=word))
        assert isinstance(a, GenAction)
        assert a.word == word

    def test_from_invalid_string(self):
        with pytest.raises(ValueError) as excinfo:
            GenAction.from_string('asdf')
        as_str = self.as_str.format(word='w')
        assert f'invalid string value for {as_str} action' in str(excinfo.value)
