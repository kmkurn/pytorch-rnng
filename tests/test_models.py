from nltk.tree import Tree
from torch.autograd import Variable
import pytest
import torch
import torch.nn as nn

from rnng.actions import ShiftAction, ReduceAction, NTAction
from rnng.models import DiscRNNG, EmptyStackError, StackLSTM, log_softmax, IllegalActionError


torch.manual_seed(12345)


class MockLSTM(object):
    def __init__(self, input_size, hidden_size, num_layers=1, **kwargs):
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


class TestStackLSTM(object):
    input_size = 10
    hidden_size = 5
    num_layers = 3
    dropout = 0.5
    seq_len = 3

    def make_stack_lstm(self, lstm_class=None):
        return StackLSTM(
            self.input_size, self.hidden_size, num_layers=self.num_layers,
            lstm_class=lstm_class
        )

    def test_init_minimal(self):
        lstm = StackLSTM(self.input_size, self.hidden_size)
        assert lstm.input_size == self.input_size
        assert lstm.hidden_size == self.hidden_size
        assert lstm.num_layers == 1
        assert lstm.dropout == pytest.approx(0, abs=1e-7)
        assert isinstance(lstm.lstm, nn.LSTM)
        assert lstm.lstm.input_size == lstm.input_size
        assert lstm.lstm.hidden_size == lstm.hidden_size
        assert lstm.lstm.num_layers == lstm.num_layers
        assert lstm.lstm.bias
        assert not lstm.lstm.batch_first
        assert lstm.lstm.dropout == pytest.approx(0, abs=1e-7)
        assert not lstm.lstm.bidirectional
        assert isinstance(lstm.h0, Variable)
        assert lstm.h0.size() == (lstm.num_layers, 1, lstm.hidden_size)
        assert isinstance(lstm.c0, Variable)
        assert lstm.c0.size() == (lstm.num_layers, 1, lstm.hidden_size)

    def test_init_full(self):
        lstm = StackLSTM(
            self.input_size, self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout, lstm_class=MockLSTM
        )
        assert lstm.num_layers == self.num_layers
        assert lstm.dropout == pytest.approx(self.dropout)
        assert isinstance(lstm.lstm, MockLSTM)

    def test_init_with_nonpositive_input_size(self):
        with pytest.raises(ValueError) as excinfo:
            StackLSTM(0, self.hidden_size)
        assert 'nonpositive input size: 0' in str(excinfo.value)

    def test_init_with_nonpositive_hidden_size(self):
        with pytest.raises(ValueError) as excinfo:
            StackLSTM(self.input_size, 0)
        assert 'nonpositive hidden size: 0' in str(excinfo.value)

    def test_init_with_nonpositive_num_layers(self):
        with pytest.raises(ValueError) as excinfo:
            StackLSTM(self.input_size, self.hidden_size, num_layers=0)
        assert 'nonpositive number of layers: 0' in str(excinfo.value)

    def test_init_with_invalid_dropout_rate(self):
        dropout = -0.1
        with pytest.raises(ValueError) as excinfo:
            StackLSTM(self.input_size, self.hidden_size, dropout=dropout)
        assert f'invalid dropout rate: {dropout}' in str(excinfo.value)

        dropout = 1.
        with pytest.raises(ValueError) as excinfo:
            StackLSTM(self.input_size, self.hidden_size, dropout=dropout)
        assert f'invalid dropout rate: {dropout}' in str(excinfo.value)

    def test_call(self):
        inputs = [Variable(torch.randn(self.input_size)) for _ in range(self.seq_len)]

        lstm = self.make_stack_lstm(lstm_class=MockLSTM)

        assert len(lstm) == 0
        h, c = lstm(inputs[0])
        assert torch.equal(h.data, lstm.lstm.retvals[0][1][0].data)
        assert torch.equal(c.data, lstm.lstm.retvals[0][1][1].data)
        assert len(lstm) == 1
        h, c = lstm(inputs[1])
        assert torch.equal(h.data, lstm.lstm.retvals[1][1][0].data)
        assert torch.equal(c.data, lstm.lstm.retvals[1][1][1].data)
        assert len(lstm) == 2
        h, c = lstm(inputs[2])
        assert torch.equal(h.data, lstm.lstm.retvals[2][1][0].data)
        assert torch.equal(c.data, lstm.lstm.retvals[2][1][1].data)
        assert len(lstm) == 3

    def test_call_with_invalid_size(self):
        lstm = self.make_stack_lstm()
        with pytest.raises(ValueError) as excinfo:
            lstm(Variable(torch.randn(2, 10)))
        assert f'expected input to have size ({lstm.input_size},), got (2, 10)' in str(
            excinfo.value
        )

    def test_top(self):
        inputs = [Variable(torch.randn(self.input_size)) for _ in range(self.seq_len)]

        lstm = self.make_stack_lstm(lstm_class=MockLSTM)

        assert lstm.top is None
        lstm(inputs[0])
        assert torch.equal(lstm.top.data, lstm.lstm.retvals[0][0].data.squeeze())
        lstm(inputs[1])
        assert torch.equal(lstm.top.data, lstm.lstm.retvals[1][0].data.squeeze())
        lstm(inputs[2])
        assert torch.equal(lstm.top.data, lstm.lstm.retvals[2][0].data.squeeze())

    def test_pop(self):
        inputs = [Variable(torch.randn(self.input_size)) for _ in range(self.seq_len)]

        lstm = self.make_stack_lstm(lstm_class=MockLSTM)
        lstm(inputs[0])
        lstm(inputs[1])
        lstm(inputs[2])

        h, c = lstm.pop()
        assert torch.equal(h.data, lstm.lstm.retvals[2][1][0].data)
        assert torch.equal(c.data, lstm.lstm.retvals[2][1][1].data)
        assert torch.equal(lstm.top.data, lstm.lstm.retvals[1][0].data.squeeze())
        assert len(lstm) == 2
        h, c = lstm.pop()
        assert torch.equal(h.data, lstm.lstm.retvals[1][1][0].data)
        assert torch.equal(c.data, lstm.lstm.retvals[1][1][1].data)
        assert torch.equal(lstm.top.data, lstm.lstm.retvals[0][0].data.squeeze())
        assert len(lstm) == 1
        h, c = lstm.pop()
        assert torch.equal(h.data, lstm.lstm.retvals[0][1][0].data)
        assert torch.equal(c.data, lstm.lstm.retvals[0][1][1].data)
        assert lstm.top is None
        assert len(lstm) == 0

    def test_pop_when_empty(self):
        lstm = self.make_stack_lstm()
        with pytest.raises(EmptyStackError):
            lstm.pop()


def test_log_softmax_without_restrictions():
    inputs = Variable(torch.randn(2, 5))

    outputs = log_softmax(inputs)

    assert isinstance(outputs, Variable)
    assert outputs.size() == inputs.size()
    assert all(x == pytest.approx(1.) for x in outputs.exp().sum(dim=1).data)


def test_log_softmax_with_restrictions():
    restrictions = torch.LongTensor([0, 2])
    inputs = Variable(torch.randn(1, 5))

    outputs = log_softmax(inputs, restrictions=restrictions)

    nonzero_indices = outputs.view(-1).exp().data.nonzero().view(-1)
    assert nonzero_indices.tolist() == [1, 3, 4]


def test_log_softmax_with_invalid_restrictions_dimension():
    restrictions = torch.LongTensor([[0, 2]])
    inputs = Variable(torch.randn(1, 5))
    with pytest.raises(ValueError) as excinfo:
        log_softmax(inputs, restrictions=restrictions)
    assert 'restrictions must have dimension of 1, got 2' in str(excinfo.value)


class TestDiscRNNG(object):
    word2id = {'John': 0, 'loves': 1, 'Mary': 2}
    pos2id = {'NNP': 0, 'VBZ': 1}
    nt2id = {'S': 2, 'NP': 1, 'VP': 0}
    actionstr2id = {'NT(S)': 0, 'NT(NP)': 1, 'NT(VP)': 2, 'SHIFT': 3, 'REDUCE': 4}

    def make_parser(self):
        return DiscRNNG(self.word2id, self.pos2id, self.nt2id, self.actionstr2id)

    def test_init_minimal(self):
        parser = DiscRNNG(self.word2id, self.pos2id, self.nt2id, self.actionstr2id)

        # Attributes
        assert parser.word2id == self.word2id
        assert parser.pos2id == self.pos2id
        assert parser.nt2id == self.nt2id
        assert parser.actionstr2id == self.actionstr2id
        assert parser.num_words == len(self.word2id)
        assert parser.num_pos == len(self.pos2id)
        assert parser.num_nt == len(self.nt2id)
        assert parser.num_actions == len(self.actionstr2id)
        assert parser.word_embedding_size == 32
        assert parser.pos_embedding_size == 12
        assert parser.nt_embedding_size == 60
        assert parser.action_embedding_size == 16
        assert parser.input_size == 128
        assert parser.hidden_size == 128
        assert parser.num_layers == 2
        assert parser.dropout == pytest.approx(0, abs=1e-7)
        assert len(parser.stack_buffer) == 0
        assert len(parser.input_buffer) == 0
        assert len(parser.action_history) == 0
        assert not parser.finished
        assert not parser.started

        # Embeddings
        assert isinstance(parser.word_embedding, nn.Embedding)
        assert parser.word_embedding.num_embeddings == parser.num_words
        assert parser.word_embedding.embedding_dim == parser.word_embedding_size
        assert isinstance(parser.pos_embedding, nn.Embedding)
        assert parser.pos_embedding.num_embeddings == parser.num_pos
        assert parser.pos_embedding.embedding_dim == parser.pos_embedding_size
        assert isinstance(parser.nt_embedding, nn.Embedding)
        assert parser.nt_embedding.num_embeddings == parser.num_nt
        assert parser.nt_embedding.embedding_dim == parser.nt_embedding_size
        assert isinstance(parser.action_embedding, nn.Embedding)
        assert parser.action_embedding.num_embeddings == parser.num_actions
        assert parser.action_embedding.embedding_dim == parser.action_embedding_size

        # Parser state encoders
        for state_name in 'stack buffer history'.split():
            state_encoder = getattr(parser, f'{state_name}_encoder')
            assert isinstance(state_encoder, StackLSTM)
            assert state_encoder.input_size == parser.input_size
            assert state_encoder.hidden_size == parser.hidden_size
            assert state_encoder.num_layers == parser.num_layers
            assert state_encoder.dropout == pytest.approx(parser.dropout, abs=1e-7)
            state_guard = getattr(parser, f'{state_name}_guard')
            assert isinstance(state_guard, nn.Parameter)
            assert state_guard.size() == (parser.input_size,)

        # Compositions
        for direction in 'fwd bwd'.split():
            composer = getattr(parser, f'{direction}_composer')
            assert isinstance(composer, nn.LSTM)
            assert composer.input_size == parser.input_size
            assert composer.hidden_size == parser.input_size
            assert composer.num_layers == parser.num_layers
            assert composer.dropout == pytest.approx(parser.dropout, abs=1e-7)
            assert composer.bias
            assert not composer.bidirectional

        # Transformation (word -> encoder)
        assert isinstance(parser.word2encoder, nn.Sequential)
        assert len(parser.word2encoder) == 2
        assert isinstance(parser.word2encoder[0], nn.Linear)
        assert parser.word2encoder[0].in_features == (
            parser.word_embedding_size + parser.pos_embedding_size
        )
        assert parser.word2encoder[0].out_features == parser.hidden_size
        assert parser.word2encoder[0].bias is not None
        assert isinstance(parser.word2encoder[1], nn.ReLU)

        # Transformation (NT -> encoder)
        assert isinstance(parser.nt2encoder, nn.Sequential)
        assert len(parser.nt2encoder) == 2
        assert isinstance(parser.nt2encoder[0], nn.Linear)
        assert parser.nt2encoder[0].in_features == parser.nt_embedding_size
        assert parser.nt2encoder[0].out_features == parser.hidden_size
        assert parser.nt2encoder[0].bias is not None
        assert isinstance(parser.nt2encoder[1], nn.ReLU)

        # Transformation (action -> encoder)
        assert isinstance(parser.action2encoder, nn.Sequential)
        assert len(parser.action2encoder) == 2
        assert isinstance(parser.action2encoder[0], nn.Linear)
        assert parser.action2encoder[0].in_features == parser.action_embedding_size
        assert parser.action2encoder[0].out_features == parser.hidden_size
        assert parser.action2encoder[0].bias is not None
        assert isinstance(parser.action2encoder[1], nn.ReLU)

        # Transformation (composer -> composed)
        assert isinstance(parser.fwdbwd2composed, nn.Sequential)
        assert len(parser.fwdbwd2composed) == 2
        assert isinstance(parser.fwdbwd2composed[0], nn.Linear)
        assert parser.fwdbwd2composed[0].in_features == 2 * parser.input_size
        assert parser.fwdbwd2composed[0].out_features == parser.input_size
        assert parser.fwdbwd2composed[0].bias is not None
        assert isinstance(parser.fwdbwd2composed[1], nn.ReLU)

        # Transformation (encoders -> parser summary)
        assert isinstance(parser.encoders2summary, nn.Sequential)
        assert len(parser.encoders2summary) == 3
        assert isinstance(parser.encoders2summary[0], nn.Dropout)
        assert parser.encoders2summary[0].p == pytest.approx(parser.dropout, abs=1e-7)
        assert isinstance(parser.encoders2summary[1], nn.Linear)
        assert parser.encoders2summary[1].in_features == 3 * parser.hidden_size
        assert parser.encoders2summary[1].out_features == parser.hidden_size
        assert parser.encoders2summary[1].bias is not None
        assert isinstance(parser.encoders2summary[2], nn.ReLU)

        # Transformation (parser summary -> action prob dist)
        assert isinstance(parser.summary2actionprobs, nn.Linear)
        assert parser.summary2actionprobs.in_features == parser.hidden_size
        assert parser.summary2actionprobs.out_features == parser.num_actions
        assert parser.summary2actionprobs.bias is not None

    def test_init_full(self):
        kwargs = dict(
            word_embedding_size=2,
            pos_embedding_size=3,
            nt_embedding_size=4,
            action_embedding_size=5,
            input_size=6,
            hidden_size=7,
            num_layers=8,
            dropout=0.5,
        )
        parser = DiscRNNG(self.word2id, self.pos2id, self.nt2id, self.actionstr2id, **kwargs)

        for key, value in kwargs.items():
            assert getattr(parser, key) == value

    def test_init_no_shift_action(self):
        actionstr2id = self.actionstr2id.copy()
        actionstr2id.pop('SHIFT')

        with pytest.raises(ValueError) as excinfo:
            DiscRNNG(self.word2id, self.pos2id, self.nt2id, actionstr2id)
        assert 'no SHIFT action found in actionstr2id mapping' in str(excinfo.value)

    def test_init_no_reduce_action(self):
        actionstr2id = self.actionstr2id.copy()
        actionstr2id.pop('REDUCE')

        with pytest.raises(ValueError) as excinfo:
            DiscRNNG(self.word2id, self.pos2id, self.nt2id, actionstr2id)
        assert 'no REDUCE action found in actionstr2id mapping' in str(excinfo.value)

    def test_start(self):
        words = 'John loves Mary'.split()
        pos_tags = 'NNP VBZ NNP'.split()
        parser = self.make_parser()

        parser.start(words, pos_tags)

        assert len(parser.stack_buffer) == 0
        assert parser.input_buffer == words
        assert len(parser.action_history) == 0
        assert not parser.finished
        assert parser.started

    def test_start_with_unequal_words_and_pos_tags_length(self):
        words = 'John loves'.split()
        pos_tags = 'NNP VBZ NNP'.split()
        parser = self.make_parser()
        with pytest.raises(ValueError) as excinfo:
            parser.start(words, pos_tags)
        assert 'words and POS tags must have equal length' in str(excinfo.value)

    def test_start_with_empty_words(self):
        parser = self.make_parser()
        with pytest.raises(ValueError) as excinfo:
            parser.start([], [])
        assert 'words cannot be empty' in str(excinfo.value)

    def test_push_nt(self):
        words = 'John loves Mary'.split()
        pos_tags = 'NNP VBZ NNP'.split()
        parser = self.make_parser()
        parser.start(words, pos_tags)
        prev_input_buffer = parser.input_buffer

        parser.push_nt('S')

        assert len(parser.stack_buffer) == 1
        last = parser.stack_buffer[-1]
        assert isinstance(last, Tree)
        assert last.label() == 'S'
        assert len(last) == 0
        assert parser.input_buffer == prev_input_buffer
        assert len(parser.action_history) == 1
        assert parser.action_history[-1] == NTAction('S')
        assert not parser.finished

    def test_illegal_push_nt(self):
        words = ['John']
        pos_tags = ['NNP']
        parser = self.make_parser()

        # Buffer is empty
        parser.start(words, pos_tags)
        parser.push_nt('S')
        parser.shift()
        with pytest.raises(IllegalActionError) as excinfo:
            parser.push_nt('NP')
        assert 'cannot do NT(X) when input buffer is empty' in str(excinfo.value)

        # More than 100 open nonterminals
        parser.start(words, pos_tags)
        for i in range(100):
            parser.push_nt('S')
        with pytest.raises(IllegalActionError) as excinfo:
            parser.push_nt('NP')
        assert 'max number of open nonterminals reached' in str(excinfo.value)

    def test_shift(self):
        words = 'John loves Mary'.split()
        pos_tags = 'NNP VBZ NNP'.split()
        parser = self.make_parser()
        parser.start(words, pos_tags)
        parser.push_nt('S')
        parser.push_nt('NP')

        parser.shift()

        assert len(parser.stack_buffer) == 3
        last = parser.stack_buffer[-1]
        assert last == 'John'
        assert parser.input_buffer == words[1:]
        assert len(parser.action_history) == 3
        assert parser.action_history[-1] == ShiftAction()
        assert not parser.finished

    def test_illegal_shift(self):
        words = ['John']
        pos_tags = ['NNP']
        parser = self.make_parser()

        # No open nonterminal
        parser.start(words, pos_tags)
        with pytest.raises(IllegalActionError) as excinfo:
            parser.shift()
        assert 'cannot SHIFT when no open nonterminal exist' in str(excinfo.value)

        # Buffer is empty
        parser.start(words, pos_tags)
        parser.push_nt('S')
        parser.shift()
        with pytest.raises(IllegalActionError) as excinfo:
            parser.shift()
        assert 'cannot SHIFT when input buffer is empty' in str(excinfo.value)

    def test_reduce(self):
        words = 'John loves Mary'.split()
        pos_tags = 'NNP VBZ NNP'.split()
        parser = self.make_parser()
        parser.start(words, pos_tags)
        parser.push_nt('S')
        parser.push_nt('NP')
        parser.shift()
        prev_input_buffer = parser.input_buffer

        parser.reduce()

        assert len(parser.stack_buffer) == 2
        last = parser.stack_buffer[-1]
        assert isinstance(last, Tree)
        assert last.label() == 'NP'
        assert len(last) == 1
        assert last[0] == 'John'
        assert parser.input_buffer == prev_input_buffer
        assert len(parser.action_history) == 4
        assert parser.action_history[-1] == ReduceAction()
        assert not parser.finished

    def test_illegal_reduce(self):
        words = 'John loves'.split()
        pos_tags = 'NNP VBZ'.split()
        parser = self.make_parser()

        # Top of stack is an open nonterminal
        parser.start(words, pos_tags)
        parser.push_nt('S')
        with pytest.raises(IllegalActionError) as excinfo:
            parser.reduce()
        assert 'cannot REDUCE when top of stack is an open nonterminal' in str(excinfo.value)

        # Buffer is not empty and REDUCE will finish parsing
        parser.start(words, pos_tags)
        parser.push_nt('S')
        parser.shift()
        with pytest.raises(IllegalActionError) as excinfo:
            parser.reduce()
        assert 'cannot REDUCE because there are words not SHIFT-ed yet' in str(excinfo.value)

    def test_do_action_when_not_started(self):
        parser = self.make_parser()

        with pytest.raises(RuntimeError) as excinfo:
            parser.push_nt('S')
        assert 'parser is not started yet, please call `start` first' in str(excinfo.value)

        with pytest.raises(RuntimeError) as excinfo:
            parser.shift()
        assert 'parser is not started yet, please call `start` first' in str(excinfo.value)

        with pytest.raises(RuntimeError) as excinfo:
            parser.reduce()
        assert 'parser is not started yet, please call `start` first' in str(excinfo.value)

    def test_forward(self):
        words = 'John loves Mary'.split()
        pos_tags = 'NNP VBZ NNP'.split()
        actions = [
            NTAction('S'),
            NTAction('NP'),
            ShiftAction(),
            ReduceAction(),
            NTAction('VP'),
            ShiftAction(),
            NTAction('NP'),
            ShiftAction(),
            ReduceAction(),
            ReduceAction(),
            ReduceAction(),
        ]
        parser = self.make_parser()

        loss = parser(words, pos_tags, actions)

        assert isinstance(loss, Variable)
        assert loss.size() == (1,)
        loss.backward()

    def test_forward_with_illegal_actions(self):
        words = 'John loves Mary'.split()
        pos_tags = 'NNP VBZ NNP'.split()
        actions = [ShiftAction()]
        parser = self.make_parser()

        loss = parser(words, pos_tags, actions)

        assert (-loss).exp().data[0] == pytest.approx(0, abs=1e-7)

    def test_finished(self):
        words = 'John loves Mary'.split()
        pos_tags = 'NNP VBZ NNP'.split()
        parser = self.make_parser()
        exp_parse_tree = Tree('S', [Tree('NP', ['John']),
                                    Tree('VP', ['loves', Tree('NP', ['Mary'])])])

        parser.start(words, pos_tags)
        parser.push_nt('S')
        parser.push_nt('NP')
        parser.shift()
        parser.reduce()
        parser.push_nt('VP')
        parser.shift()
        parser.push_nt('NP')
        parser.shift()
        parser.reduce()
        parser.reduce()
        parser.reduce()

        assert parser.finished
        parse_tree = parser.stack_buffer[-1]
        assert parse_tree == exp_parse_tree

        with pytest.raises(RuntimeError) as excinfo:
            parser.push_nt('NP')
        assert 'cannot do action when parser is finished' in str(excinfo.value)

        with pytest.raises(RuntimeError) as excinfo:
            parser.shift()
        assert 'cannot do action when parser is finished' in str(excinfo.value)

        with pytest.raises(RuntimeError) as excinfo:
            parser.reduce()
        assert 'cannot do action when parser is finished' in str(excinfo.value)
