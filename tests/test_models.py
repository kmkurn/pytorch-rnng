import pytest

from rnng.models import StackLSTM, EmptyStackError


class MockLSTMCell:
    def __init__(self):
        self.index = 0
        self.retval = [('a', 1), ('b', 2), ('c', 3)]

    def __call__(self, *args, **kwargs):
        res = self.retval[self.index]
        self.index = (self.index + 1) % len(self.retval)
        return res


class TestStackLSTM:
    def test_call(self, mocker):
        mock_cell = MockLSTMCell()
        mocker.patch('rnng.models.LSTMCell', return_value=mock_cell)
        lstm = StackLSTM(10, 5)
        assert lstm('asdf') == mock_cell.retval[0]
        assert lstm('fdsa') == mock_cell.retval[1]
        assert lstm('aass') == mock_cell.retval[2]

    def test_top(self, mocker):
        mock_cell = MockLSTMCell()
        mocker.patch('rnng.models.LSTMCell', return_value=mock_cell)
        lstm = StackLSTM(10, 5)
        assert lstm.top is None
        lstm('asdf')
        assert lstm.top == mock_cell.retval[0][0]
        lstm('fdsa')
        assert lstm.top == mock_cell.retval[1][0]
        lstm('aass')
        assert lstm.top == mock_cell.retval[2][0]

    def test_pop(self, mocker):
        mock_cell = MockLSTMCell()
        mocker.patch('rnng.models.LSTMCell', return_value=mock_cell)
        lstm = StackLSTM(10, 5)
        lstm('asdf')
        lstm('fdsa')
        lstm('aass')
        assert lstm.pop() == mock_cell.retval[2]
        assert lstm.top == mock_cell.retval[1][0]
        assert lstm.pop() == mock_cell.retval[1]
        assert lstm.top == mock_cell.retval[0][0]
        assert lstm.pop() == mock_cell.retval[0]
        assert lstm.top is None

    def test_pop_when_empty(self, mocker):
        mocker.patch('rnng.models.LSTMCell', return_value=MockLSTMCell())
        lstm = StackLSTM(10, 5)
        with pytest.raises(EmptyStackError):
            lstm.pop()
