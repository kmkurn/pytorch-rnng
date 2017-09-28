import pytest
import torch
from torch.autograd import Variable

from rnng.models import StackedLSTMCell, StackLSTM, EmptyStackError


class MockLSTMCell:
    def __init__(self, retval=None):
        if retval is None:
            retval = [('a', 1), ('b', 2), ('c', 3)]

        self.index = 0
        self.retval = retval

    def __call__(self, *args, **kwargs):
        res = self.retval[self.index]
        self.index = (self.index + 1) % len(self.retval)
        return res


class TestStackedLSTMCell:
    def test_call(self, mocker):
        batch_size = 3
        input_size = 10
        hidden_size = 5
        num_layers = 3
        cells = [MockLSTMCell(retval=[(Variable(torch.randn(batch_size, hidden_size)),
                                       Variable(torch.randn(batch_size, hidden_size)))
                                      for _ in range(3)])
                 for i in range(num_layers)]
        mocker.patch('rnng.models.LSTMCell', side_effect=cells)

        inputs = Variable(torch.randn(batch_size, input_size))
        h0 = Variable(torch.randn(num_layers, batch_size, hidden_size))
        c0 = Variable(torch.randn(num_layers, batch_size, hidden_size))
        expected_h1 = torch.stack([c.retval[0][0] for c in cells])
        expected_c1 = torch.stack([c.retval[0][1] for c in cells])

        stacked_cell = StackedLSTMCell(input_size, hidden_size, num_layers=num_layers)
        h1, c1 = stacked_cell(inputs, (h0, c0))

        assert torch.equal(h1.data, expected_h1.data)
        assert torch.equal(c1.data, expected_c1.data)

    def test_num_layers_too_low(self):
        with pytest.raises(ValueError):
            StackedLSTMCell(10, 5, num_layers=0)

    def test_init_states_fewer_than_layers(self):
        batch_size = 4
        input_size = 10
        hidden_size = 5
        stacked_cell = StackedLSTMCell(input_size, hidden_size, num_layers=3)
        inputs = Variable(torch.randn(batch_size, input_size))
        h0 = Variable(torch.randn(2, batch_size, hidden_size))
        c0 = Variable(torch.randn(2, batch_size, hidden_size))

        with pytest.raises(ValueError):
            stacked_cell(inputs, (h0, c0))


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
