from nltk.tree import Tree
import pytest
import torch
from torch.autograd import Variable

from rnng.models import DiscRNNGrammar, EmptyStackError, StackLSTM, log_softmax


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


def test_log_softmax():
    restrictions = torch.LongTensor([0, 2])
    inputs = Variable(torch.randn(1, 5))

    outputs = log_softmax(inputs, restrictions)

    assert isinstance(outputs, Variable)
    assert outputs.size() == (1, 5)
    nonzero_indices = outputs.view(-1).exp().data.nonzero().view(-1)
    assert all(nonzero_indices.eq(torch.LongTensor([1, 3, 4])))


class TestDiscRNNGrammar:
    word2id = {'John': 0, 'loves': 1, 'Mary': 2}
    pos2id = {'NNP': 0, 'VBZ': 1}
    nt2id = {'S': 0, 'NP': 1, 'VP': 2}
    action2id = {'NT(S)': 0, 'NT(NP)': 1, 'NT(VP)': 2, 'SHIFT': 3, 'REDUCE': 4}
    action2nt = {0: 0, 1: 1, 2: 2}

    def test_init(self):
        parser = DiscRNNGrammar(
            len(self.word2id), len(self.pos2id), len(self.nt2id), len(self.action2id),
            self.action2id['SHIFT'], self.action2nt)

        assert len(parser.stack_buffer) == 0
        assert len(parser.input_buffer) == 0
        assert len(parser.action_history) == 0
        assert parser.num_open_nt == 0
        assert not parser.finished

    def test_start(self):
        words = [self.word2id[w] for w in ['John', 'loves', 'Mary']]
        pos_tags = [self.pos2id[p] for p in ['NNP', 'VBZ', 'NNP']]
        parser = DiscRNNGrammar(
            len(self.word2id), len(self.pos2id), len(self.nt2id), len(self.action2id),
            self.action2id['SHIFT'], self.action2nt)

        parser.start(zip(words, pos_tags))

        assert len(parser.stack_buffer) == 0
        assert parser.input_buffer == tuple(words)
        assert len(parser.action_history) == 0
        assert parser.num_open_nt == 0
        assert not parser.finished

    def test_do_nt_action(self):
        words = [self.word2id[w] for w in ['John', 'loves', 'Mary']]
        pos_tags = [self.pos2id[p] for p in ['NNP', 'VBZ', 'NNP']]
        parser = DiscRNNGrammar(
            len(self.word2id), len(self.pos2id), len(self.nt2id), len(self.action2id),
            self.action2id['SHIFT'], self.action2nt)
        parser.start(zip(words, pos_tags))
        prev_input_buffer = parser.input_buffer

        parser.do_action(self.action2id['NT(S)'])

        assert len(parser.stack_buffer) == 1
        last = parser.stack_buffer[-1]
        assert isinstance(last.subtree, Tree)
        assert last.subtree.label() == self.nt2id['S']
        assert last.is_open_nt
        assert parser.input_buffer == prev_input_buffer
        assert len(parser.action_history) == 1
        assert parser.action_history[-1] == self.action2id['NT(S)']
        assert parser.num_open_nt == 1
        assert not parser.finished

    def test_do_shift_action(self):
        words = [self.word2id[w] for w in ['John', 'loves', 'Mary']]
        pos_tags = [self.pos2id[p] for p in ['NNP', 'VBZ', 'NNP']]
        parser = DiscRNNGrammar(
            len(self.word2id), len(self.pos2id), len(self.nt2id), len(self.action2id),
            self.action2id['SHIFT'], self.action2nt)
        parser.start(zip(words, pos_tags))
        parser.do_action(self.action2id['NT(S)'])
        parser.do_action(self.action2id['NT(NP)'])
        prev_num_open_nt = parser.num_open_nt

        parser.do_action(self.action2id['SHIFT'])

        assert len(parser.stack_buffer) == 3
        last = parser.stack_buffer[-1]
        assert last.subtree == self.word2id['John']
        assert not last.is_open_nt
        assert parser.input_buffer == tuple(words[1:])
        assert len(parser.action_history) == 3
        assert parser.action_history[-1] == self.action2id['SHIFT']
        assert parser.num_open_nt == prev_num_open_nt
        assert not parser.finished

    def test_do_reduce_action(self):
        words = [self.word2id[w] for w in ['John', 'loves', 'Mary']]
        pos_tags = [self.pos2id[p] for p in ['NNP', 'VBZ', 'NNP']]
        parser = DiscRNNGrammar(
            len(self.word2id), len(self.pos2id), len(self.nt2id), len(self.action2id),
            self.action2id['SHIFT'], self.action2nt)
        parser.start(zip(words, pos_tags))
        parser.do_action(self.action2id['NT(S)'])
        parser.do_action(self.action2id['NT(NP)'])
        parser.do_action(self.action2id['SHIFT'])
        prev_input_buffer = parser.input_buffer
        prev_num_open_nt = parser.num_open_nt

        parser.do_action(self.action2id['REDUCE'])

        assert len(parser.stack_buffer) == 2
        last = parser.stack_buffer[-1]
        assert isinstance(last.subtree, Tree)
        assert last.subtree.label() == self.nt2id['NP']
        assert len(last.subtree) == 1
        assert last.subtree[0] == self.word2id['John']
        assert not last.is_open_nt
        assert parser.input_buffer == prev_input_buffer
        assert len(parser.action_history) == 4
        assert parser.action_history[-1] == self.action2id['REDUCE']
        assert parser.num_open_nt == prev_num_open_nt - 1
        assert not parser.finished

    def test_forward(self):
        words = [self.word2id[w] for w in ['John', 'loves', 'Mary']]
        pos_tags = [self.pos2id[p] for p in ['NNP', 'VBZ', 'NNP']]
        parser = DiscRNNGrammar(
            len(self.word2id), len(self.pos2id), len(self.nt2id), len(self.action2id),
            self.action2id['SHIFT'], self.action2nt)
        parser.start(zip(words, pos_tags))
        parser.do_action(self.action2id['NT(S)'])
        parser.do_action(self.action2id['NT(NP)'])
        parser.do_action(self.action2id['SHIFT'])
        parser.do_action(self.action2id['REDUCE'])

        action_logprobs = parser()

        assert isinstance(action_logprobs, Variable)
        assert action_logprobs.size() == (len(self.action2id),)
        sum_prob = action_logprobs.exp().sum().data[0]
        assert 0.999 <= sum_prob and sum_prob <= 1.001

    def test_finished(self):
        words = [self.word2id[w] for w in ['John', 'loves', 'Mary']]
        pos_tags = [self.pos2id[p] for p in ['NNP', 'VBZ', 'NNP']]
        parser = DiscRNNGrammar(
            len(self.word2id), len(self.pos2id), len(self.nt2id), len(self.action2id),
            self.action2id['SHIFT'], self.action2nt)
        exp_parse_tree = Tree(self.nt2id['S'], [
            Tree(self.nt2id['NP'], [
                self.word2id['John']]),
            Tree(self.nt2id['VP'], [
                self.word2id['loves'],
                Tree(self.nt2id['NP'], [
                    self.word2id['Mary']
                ])
            ])
        ])

        parser.start(zip(words, pos_tags))
        parser.do_action(self.action2id['NT(S)'])
        parser.do_action(self.action2id['NT(NP)'])
        parser.do_action(self.action2id['SHIFT'])
        parser.do_action(self.action2id['REDUCE'])
        parser.do_action(self.action2id['NT(VP)'])
        parser.do_action(self.action2id['SHIFT'])
        parser.do_action(self.action2id['NT(NP)'])
        parser.do_action(self.action2id['SHIFT'])
        parser.do_action(self.action2id['REDUCE'])
        parser.do_action(self.action2id['REDUCE'])
        parser.do_action(self.action2id['REDUCE'])

        assert parser.finished
        parse_tree = parser.stack_buffer[-1].subtree
        assert str(parse_tree) == str(exp_parse_tree)
        with pytest.raises(RuntimeError):
            parser.do_action(self.action2id['REDUCE'])
