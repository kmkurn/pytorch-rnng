from unittest.mock import Mock

import pytest

from rnng.actions import ShiftAction, ReduceAction, NTAction, GenAction


class TestShiftAction:
    as_str = 'SHIFT'

    @staticmethod
    def make_action():
        return ShiftAction()

    def test_eq(self):
        a1 = self.make_action()
        a2 = self.make_action()
        assert a1 == a2
        assert a1 != 'foo'

    def test_hash(self):
        a = self.make_action()
        assert hash(a) == hash(self.as_str)

    def test_str(self):
        a = self.make_action()
        assert str(a) == self.as_str

    def test_execute_on(self):
        a = self.make_action()
        fake_parser = Mock()
        a.execute_on(fake_parser)
        fake_parser.shift.assert_called_once_with()

    def test_from_string(self):
        a = ShiftAction.from_string(self.as_str)
        assert isinstance(a, ShiftAction)

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            ShiftAction.from_string('asdf')

    def test_verify_on(self):
        a = self.make_action()
        fake_parser = Mock()
        a.verify_on(fake_parser)
        fake_parser.verify_shift.assert_called_once_with()


class TestReduceAction:
    as_str = 'REDUCE'

    @staticmethod
    def make_action():
        return ReduceAction()

    def test_eq(self):
        a1 = self.make_action()
        a2 = self.make_action()
        assert a1 == a2
        assert ReduceAction() != 'foo'

    def test_hash(self):
        a = self.make_action()
        assert hash(a) == hash(self.as_str)

    def test_str(self):
        a = self.make_action()
        assert str(a) == self.as_str

    def test_execute_on(self):
        a = self.make_action()
        fake_parser = Mock()
        a.execute_on(fake_parser)
        fake_parser.reduce.assert_called_once_with()

    def test_from_string(self):
        a = ReduceAction.from_string(self.as_str)
        assert isinstance(a, ReduceAction)

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            ReduceAction.from_string('asdf')

    def test_verify_on(self):
        a = self.make_action()
        fake_parser = Mock()
        a.verify_on(fake_parser)
        fake_parser.verify_reduce.assert_called_once_with()


class TestNTAction:
    as_str = 'NT({label})'

    @staticmethod
    def make_action(label='NP'):
        return NTAction(label)

    def test_eq(self):
        a1 = self.make_action()
        a2 = self.make_action()
        a3 = self.make_action('VP')
        assert a1 == a2
        assert a1 != a3
        assert a1 != 'foo'

    def test_hash(self):
        a = self.make_action()
        assert hash(a) == hash(self.as_str.format(label=a.label))

    def test_str(self):
        label = 'NP'
        a = self.make_action(label=label)
        assert str(a) == self.as_str.format(label=label)

    def test_execute_on(self):
        a = self.make_action()
        fake_parser = Mock()
        a.execute_on(fake_parser)
        fake_parser.push_nt.assert_called_once_with(a.label)

    def test_from_string(self):
        label = 'NP'
        a = NTAction.from_string(self.as_str.format(label=label))
        assert isinstance(a, NTAction)
        assert a.label == label

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            NTAction.from_string('asdf')

    def test_verify_on(self):
        a = self.make_action()
        fake_parser = Mock()
        a.verify_on(fake_parser)
        fake_parser.verify_push_nt.assert_called_once_with()


class TestGenAction:
    as_str = 'GEN({word})'

    @staticmethod
    def make_action(word='asdf'):
        return GenAction(word)

    def test_eq(self):
        a1 = self.make_action()
        a2 = self.make_action()
        a3 = self.make_action(word='fdsa')
        assert a1 == a2
        assert a1 != a3
        assert a1 != 'foo'

    def test_hash(self):
        word = 'asdf'
        a = self.make_action(word=word)
        assert hash(a) == hash(self.as_str.format(word=word))

    def test_str(self):
        word = 'asdf'
        a = self.make_action(word=word)
        assert str(a) == self.as_str.format(word=word)

    def test_from_string(self):
        word = 'asdf'
        a = GenAction.from_string(self.as_str.format(word=word))
        assert isinstance(a, GenAction)
        assert a.word == word

    def test_from_invalid_string(self):
        with pytest.raises(ValueError):
            GenAction.from_string('asdf')
