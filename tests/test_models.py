import pytest
import torch
from torch.autograd import Variable

from rnng.models import StackedLSTMCell, StackLSTM, EmptyStackError


class MockLSTMCell:
    def __init__(self, retvals=None):
        if retvals is None:
            retvals = [('a', 1), ('b', 2), ('c', 3)]

        self.index = 0
        self.retvals = retvals

    def __call__(self, *args, **kwargs):
        res = self.retvals[self.index]
        self.index = (self.index + 1) % len(self.retvals)
        return res


class TestStackedLSTMCell:
    def test_call(self, mocker):
        batch_size = 3
        input_size = 10
        hidden_size = 5
        num_layers = 3
        cells = [MockLSTMCell(retvals=[(Variable(torch.randn(batch_size, hidden_size)),
                                        Variable(torch.randn(batch_size, hidden_size)))
                                       for _ in range(3)])
                 for i in range(num_layers)]
        mocker.patch('rnng.models.LSTMCell', side_effect=cells)

        inputs = Variable(torch.randn(batch_size, input_size))
        h0 = Variable(torch.randn(num_layers, batch_size, hidden_size))
        c0 = Variable(torch.randn(num_layers, batch_size, hidden_size))
        expected_h1 = torch.stack([c.retvals[0][0] for c in cells])
        expected_c1 = torch.stack([c.retvals[0][1] for c in cells])

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
    batch_size = 1
    input_size = 10
    hidden_size = 5
    num_layers = 3
    seq_len = 3

    def test_call(self, mocker):
        mock_cell = MockLSTMCell(retvals=[(Variable(torch.randn(self.num_layers,
                                                                self.batch_size,
                                                                self.hidden_size)),
                                           Variable(torch.randn(self.num_layers,
                                                                self.batch_size,
                                                                self.hidden_size)))
                                          for _ in range(3)])
        mocker.patch('rnng.models.StackedLSTMCell', return_value=mock_cell)

        lstm = StackLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        inputs = [Variable(torch.randn(self.batch_size, self.input_size))
                  for _ in range(self.seq_len)]

        h, c = lstm(inputs[0])
        assert torch.equal(h.data, mock_cell.retvals[0][0].data)
        assert torch.equal(c.data, mock_cell.retvals[0][1].data)
        h, c = lstm(inputs[1])
        assert torch.equal(h.data, mock_cell.retvals[1][0].data)
        assert torch.equal(c.data, mock_cell.retvals[1][1].data)
        h, c = lstm(inputs[2])
        assert torch.equal(h.data, mock_cell.retvals[2][0].data)
        assert torch.equal(c.data, mock_cell.retvals[2][1].data)

    def test_top(self, mocker):
        mock_cell = MockLSTMCell(retvals=[(Variable(torch.randn(self.num_layers,
                                                                self.batch_size,
                                                                self.hidden_size)),
                                           Variable(torch.randn(self.num_layers,
                                                                self.batch_size,
                                                                self.hidden_size)))
                                          for _ in range(3)])
        mocker.patch('rnng.models.StackedLSTMCell', return_value=mock_cell)

        lstm = StackLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        inputs = [Variable(torch.randn(self.batch_size, self.input_size))
                  for _ in range(self.seq_len)]

        assert lstm.top is None
        lstm(inputs[0])
        assert torch.equal(lstm.top.data, mock_cell.retvals[0][0][-1].data)
        lstm(inputs[1])
        assert torch.equal(lstm.top.data, mock_cell.retvals[1][0][-1].data)
        lstm(inputs[2])
        assert torch.equal(lstm.top.data, mock_cell.retvals[2][0][-1].data)

    def test_pop(self, mocker):
        mock_cell = MockLSTMCell(retvals=[(Variable(torch.randn(self.num_layers,
                                                                self.batch_size,
                                                                self.hidden_size)),
                                           Variable(torch.randn(self.num_layers,
                                                                self.batch_size,
                                                                self.hidden_size)))
                                          for _ in range(3)])
        mocker.patch('rnng.models.StackedLSTMCell', return_value=mock_cell)

        lstm = StackLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        inputs = [Variable(torch.randn(self.batch_size, self.input_size))
                  for _ in range(self.seq_len)]

        lstm(inputs[0])
        lstm(inputs[1])
        lstm(inputs[2])

        h, c = lstm.pop()
        assert torch.equal(h.data, mock_cell.retvals[2][0].data)
        assert torch.equal(c.data, mock_cell.retvals[2][1].data)
        assert torch.equal(lstm.top.data, mock_cell.retvals[1][0][-1].data)
        h, c = lstm.pop()
        assert torch.equal(h.data, mock_cell.retvals[1][0].data)
        assert torch.equal(c.data, mock_cell.retvals[1][1].data)
        assert torch.equal(lstm.top.data, mock_cell.retvals[0][0][-1].data)
        h, c = lstm.pop()
        assert torch.equal(h.data, mock_cell.retvals[0][0].data)
        assert torch.equal(c.data, mock_cell.retvals[0][1].data)
        assert lstm.top is None

    def test_pop_when_empty(self, mocker):
        mocker.patch('rnng.models.StackedLSTMCell', return_value=MockLSTMCell())
        lstm = StackLSTM(10, 5, num_layers=2)
        with pytest.raises(EmptyStackError):
            lstm.pop()

    def test_num_layers_too_low(self):
        with pytest.raises(ValueError):
            StackLSTM(10, 5, num_layers=0)
