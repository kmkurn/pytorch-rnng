from nltk.tree import Tree
from torch.autograd import Variable
import pytest
import torch
import torch.nn as nn

from rnng.actions import NT, REDUCE, SHIFT, get_nonterm
from rnng.models import DiscRNNG, EmptyStackError, StackLSTM, log_softmax


torch.manual_seed(12345)


class MockLSTM(object):
    def __init__(self, input_size, hidden_size, num_layers=1, **kwargs):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.index = 0
        self.retvals = [(self._get_output(), self._get_hn_cn()) for _ in range(3)]

    def named_parameters(self):
        return []

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
    nt2id = {'S': 0, 'NP': 1, 'VP': 2}
    num_words = len(word2id)
    num_pos = len(pos2id)
    num_nt = len(nt2id)

    def make_parser(self):
        return DiscRNNG(
            self.num_words, self.num_pos, self.num_nt)

    def make_words(self, words=None):
        if words is None:
            words = 'John loves Mary'.split()
        return Variable(torch.LongTensor([self.word2id[x] for x in words]))

    def make_pos_tags(self, pos_tags=None):
        if pos_tags is None:
            pos_tags = 'NNP VBZ NNP'.split()
        return Variable(torch.LongTensor([self.pos2id[x] for x in pos_tags]))

    def make_actions(self, actions=None):
        if actions is None:
            actions = [
                NT('S'),
                NT('NP'),
                SHIFT,
                REDUCE,
                NT('VP'),
                SHIFT,
                NT('NP'),
                SHIFT,
                REDUCE,
                REDUCE,
                REDUCE,
            ]

        return Variable(torch.LongTensor([self.action2id(x) for x in actions]))

    def action2id(self, action):
        if action == REDUCE:
            return 0
        if action == SHIFT:
            return 1
        return self.nt2id[get_nonterm(action)] + 2

    def test_init_minimal(self):
        parser = DiscRNNG(
            self.num_words, self.num_pos, self.num_nt)

        # Attributes
        assert parser.num_words == self.num_words
        assert parser.num_pos == self.num_pos
        assert parser.num_nt == self.num_nt
        assert parser.num_actions == self.num_nt + 2
        assert parser.word_embedding_size == 32
        assert parser.pos_embedding_size == 12
        assert parser.nt_embedding_size == 60
        assert parser.action_embedding_size == 16
        assert parser.input_size == 128
        assert parser.hidden_size == 128
        assert parser.num_layers == 2
        assert parser.dropout == pytest.approx(0, abs=1e-7)
        assert not parser.finished

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
        assert isinstance(parser.summary2actionlogprobs, nn.Linear)
        assert parser.summary2actionlogprobs.in_features == parser.hidden_size
        assert parser.summary2actionlogprobs.out_features == parser.num_actions
        assert parser.summary2actionlogprobs.bias is not None

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
        parser = DiscRNNG(
            self.num_words, self.num_pos, self.num_nt, **kwargs)

        for key, value in kwargs.items():
            assert getattr(parser, key) == value

    def test_forward(self):
        words = self.make_words()
        pos_tags = self.make_pos_tags()
        actions = self.make_actions()
        parser = self.make_parser()

        llh = parser(words, pos_tags, actions)

        assert isinstance(llh, Variable)
        assert llh.size() == (1,)
        llh.backward()
        assert parser.finished

    def test_forward_with_shift_when_buffer_is_empty(self):
        words = self.make_words()
        pos_tags = self.make_pos_tags()
        actions = self.make_actions([
            NT('S'), SHIFT, SHIFT, SHIFT, SHIFT])
        parser = self.make_parser()
        llh = parser(words, pos_tags, actions)
        assert llh.exp().data[0] == pytest.approx(0, abs=1e-7)

    def test_forward_with_shift_when_no_open_nt_in_the_stack(self):
        words = self.make_words()
        pos_tags = self.make_pos_tags()
        actions = self.make_actions([SHIFT])
        parser = self.make_parser()
        llh = parser(words, pos_tags, actions)
        assert llh.exp().data[0] == pytest.approx(0, abs=1e-7)

    def test_forward_with_reduce_when_tos_is_an_open_nt(self):
        words = self.make_words()
        pos_tags = self.make_pos_tags()
        actions = self.make_actions([NT('S'), REDUCE])
        parser = self.make_parser()
        llh = parser(words, pos_tags, actions)
        assert llh.exp().data[0] == pytest.approx(0, abs=1e-7)

    def test_forward_with_reduce_when_only_single_open_nt_and_buffer_is_not_empty(self):
        words = self.make_words()
        pos_tags = self.make_pos_tags()
        actions = self.make_actions([NT('S'), SHIFT, REDUCE])
        parser = self.make_parser()
        llh = parser(words, pos_tags, actions)
        assert llh.exp().data[0] == pytest.approx(0, abs=1e-7)

    def test_forward_with_push_nt_when_buffer_is_empty(self):
        words = self.make_words()
        pos_tags = self.make_pos_tags()
        actions = self.make_actions([
            NT('S'), SHIFT, SHIFT, SHIFT, NT('NP')])
        parser = self.make_parser()
        llh = parser(words, pos_tags, actions)
        assert llh.exp().data[0] == pytest.approx(0, abs=1e-7)

    def test_forward_with_push_nt_when_maximum_number_of_open_nt_is_reached(self):
        DiscRNNG.MAX_OPEN_NT = 2
        words = self.make_words()
        pos_tags = self.make_pos_tags()
        actions = self.make_actions([NT('S')] * (DiscRNNG.MAX_OPEN_NT+1))
        parser = self.make_parser()
        llh = parser(words, pos_tags, actions)
        assert llh.exp().data[0] == pytest.approx(0, abs=1e-7)

    def test_forward_with_bad_dimensions(self):
        words = Variable(torch.randn(2, 3)).long()
        pos_tags = Variable(torch.randn(3)).long()
        actions = Variable(torch.randn(5)).long()
        parser = self.make_parser()
        with pytest.raises(ValueError) as excinfo:
            parser(words, pos_tags, actions)
        assert 'expected words to have dimension of 1, got 2' in str(excinfo.value)

        words = Variable(torch.randn(3)).long()
        pos_tags = Variable(torch.randn(2, 3)).long()
        with pytest.raises(ValueError) as excinfo:
            parser(words, pos_tags, actions)
        assert 'expected POS tags to have size equal to words' in str(excinfo.value)

        words = Variable(torch.randn(3)).long()
        pos_tags = Variable(torch.randn(3)).long()
        actions = Variable(torch.randn(5, 3)).long()
        with pytest.raises(ValueError) as excinfo:
            parser(words, pos_tags, actions)
        assert 'expected actions to have dimension of 1, got 2' in str(excinfo.value)

    def test_decode(self):
        words = self.make_words()
        pos_tags = self.make_pos_tags()
        parser = self.make_parser()

        best_action_ids, parse_tree = parser.decode(words, pos_tags)

        assert isinstance(best_action_ids, list)
        assert isinstance(parse_tree, Tree)
        assert parser.finished
