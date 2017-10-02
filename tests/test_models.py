import pytest
import torch
from torch.autograd import Variable

from rnng.models import StackLSTM, EmptyStackError


class MockLSTM:
    def __init__(self, input_size, hidden_size, num_layers=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.index = 0
        self.retvals = [(self._get_output(), self._get_hn_cn()) for _ in range(3)]

    def __call__(self, inputs, init_states):
        retval = self.retvals[self.index]
        self.index = (self.index + 1) % len(self.retvals)
        return retval

    def _get_output(self):
        return Variable(torch.randn(1, 1, self.hidden_size))

    def _get_hn_cn(self):
        return (Variable(torch.randn(self.num_layers, 1, self.hidden_size)),
                Variable(torch.randn(self.num_layers, 1, self.hidden_size)))


class TestStackLSTM:
    input_size = 10
    hidden_size = 5
    num_layers = 3
    seq_len = 3

    def test_call(self, mocker):
        mock_lstm = MockLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        mocker.patch('rnng.models.nn.LSTM', return_value=mock_lstm, autospec=True)
        inputs = [Variable(torch.randn(self.input_size)) for _ in range(self.seq_len)]

        lstm = StackLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)

        assert len(lstm) == 0
        h, c = lstm(inputs[0])
        assert torch.equal(h.data, mock_lstm.retvals[0][1][0].data)
        assert torch.equal(c.data, mock_lstm.retvals[0][1][1].data)
        assert len(lstm) == 1
        h, c = lstm(inputs[1])
        assert torch.equal(h.data, mock_lstm.retvals[1][1][0].data)
        assert torch.equal(c.data, mock_lstm.retvals[1][1][1].data)
        assert len(lstm) == 2
        h, c = lstm(inputs[2])
        assert torch.equal(h.data, mock_lstm.retvals[2][1][0].data)
        assert torch.equal(c.data, mock_lstm.retvals[2][1][1].data)
        assert len(lstm) == 3

    def test_top(self, mocker):
        mock_lstm = MockLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        mocker.patch('rnng.models.nn.LSTM', return_value=mock_lstm, autospec=True)
        inputs = [Variable(torch.randn(self.input_size)) for _ in range(self.seq_len)]

        lstm = StackLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)

        assert lstm.top is None
        lstm(inputs[0])
        assert torch.equal(lstm.top.data, mock_lstm.retvals[0][0].data.squeeze())
        lstm(inputs[1])
        assert torch.equal(lstm.top.data, mock_lstm.retvals[1][0].data.squeeze())
        lstm(inputs[2])
        assert torch.equal(lstm.top.data, mock_lstm.retvals[2][0].data.squeeze())

    def test_pop(self, mocker):
        mock_lstm = MockLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        mocker.patch('rnng.models.nn.LSTM', return_value=mock_lstm, autospec=True)
        inputs = [Variable(torch.randn(self.input_size)) for _ in range(self.seq_len)]

        lstm = StackLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        lstm(inputs[0])
        lstm(inputs[1])
        lstm(inputs[2])

        h, c = lstm.pop()
        assert torch.equal(h.data, mock_lstm.retvals[2][1][0].data)
        assert torch.equal(c.data, mock_lstm.retvals[2][1][1].data)
        assert torch.equal(lstm.top.data, mock_lstm.retvals[1][0].data.squeeze())
        assert len(lstm) == 2
        h, c = lstm.pop()
        assert torch.equal(h.data, mock_lstm.retvals[1][1][0].data)
        assert torch.equal(c.data, mock_lstm.retvals[1][1][1].data)
        assert torch.equal(lstm.top.data, mock_lstm.retvals[0][0].data.squeeze())
        assert len(lstm) == 1
        h, c = lstm.pop()
        assert torch.equal(h.data, mock_lstm.retvals[0][1][0].data)
        assert torch.equal(c.data, mock_lstm.retvals[0][1][1].data)
        assert lstm.top is None
        assert len(lstm) == 0

    def test_pop_when_empty(self, mocker):
        mock_lstm = MockLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        mocker.patch('rnng.models.nn.LSTM', return_value=mock_lstm, autospec=True)

        lstm = StackLSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)
        with pytest.raises(EmptyStackError):
            lstm.pop()

    def test_num_layers_too_low(self):
        with pytest.raises(ValueError):
            StackLSTM(10, 5, num_layers=0)
