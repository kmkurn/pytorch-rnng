import abc
from collections import OrderedDict
from typing import Collection, List, Mapping, NamedTuple, Sequence, Sized, Tuple, Union
from typing import Dict  # noqa

from nltk.tree import Tree
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from rnng.typing import WordId, POSId, NTId, ActionId


class EmptyStackError(Exception):
    def __init__(self):
        super().__init__('stack is already empty')


class StackLSTM(nn.Module, Sized):
    BATCH_SIZE = 1
    SEQ_LEN = 1

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.) -> None:
        if num_layers < 1:
            raise ValueError('number of layers is at least 1')

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.h0 = nn.Parameter(torch.Tensor(num_layers, self.BATCH_SIZE, hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(num_layers, self.BATCH_SIZE, hidden_size))
        init_states = (self.h0, self.c0)
        self._states_hist = [init_states]
        self._outputs_hist = []  # type: List[Variable]

    def forward(self, inputs: Variable) -> Tuple[Variable, Variable]:
        # inputs: input_size
        assert self._states_hist

        # Set seq_len and batch_size to 1
        inputs = inputs.view(self.SEQ_LEN, self.BATCH_SIZE, inputs.numel())
        next_outputs, next_states = self.lstm(inputs, self._states_hist[-1])
        self._states_hist.append(next_states)
        self._outputs_hist.append(next_outputs)
        return next_states

    def push(self, *args, **kwargs):
        return self(*args, **kwargs)

    def pop(self) -> Tuple[Variable, Variable]:
        if len(self._states_hist) > 1:
            self._outputs_hist.pop()
            return self._states_hist.pop()
        else:
            raise EmptyStackError()

    @property
    def top(self) -> Variable:
        # outputs: hidden_size
        return self._outputs_hist[-1].squeeze() if self._outputs_hist else None

    def __repr__(self) -> str:
        res = ('{}(input_size={input_size}, hidden_size={hidden_size}, '
               'num_layers={num_layers}, dropout={dropout})')
        return res.format(self.__class__.__name__, **self.__dict__)

    def __len__(self):
        return len(self._outputs_hist)


def log_softmax(inputs: Variable, restrictions=None) -> Variable:
    if restrictions is None:
        return F.log_softmax(inputs)

    if restrictions.dim() != 1:
        raise RuntimeError('restrictions must be one dimensional')

    addend = Variable(
        inputs.data.new(inputs.size()).zero_().index_fill_(
            inputs.dim() - 1, restrictions, -float('inf')))
    return F.log_softmax(inputs + addend)


class StackElement(NamedTuple):
    subtree: Union[WordId, Tree]
    emb: Variable
    is_open_nt: bool


class RNNGrammar(nn.Module):
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def stack_buffer(self) -> Sequence[Union[Tree, WordId]]:
        pass

    @property
    @abc.abstractmethod
    def action_history(self) -> Sequence[ActionId]:
        pass

    @property
    @abc.abstractmethod
    def finished(self) -> bool:
        pass

    @abc.abstractmethod
    def start(self, tagged_words: Sequence[Tuple[WordId, POSId]]) -> None:
        pass

    @abc.abstractmethod
    def do_action(self, action: ActionId) -> None:
        pass


class DiscRNNGrammar(RNNGrammar):
    MAX_OPEN_NT = 100

    def __init__(self, num_words: int, num_pos: int, num_nt: int, num_actions: int,
                 shift_action: ActionId, action2nt: Mapping[ActionId, NTId],
                 word_dim: int = 32, pos_dim: int = 12, nt_dim: int = 60, action_dim: int = 16,
                 input_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.) -> None:
        if shift_action < 0 or shift_action >= num_actions:
            raise ValueError('SHIFT action ID is out of range')
        for action, nonterm in action2nt.items():
            if action < 0 or action >= num_actions:
                raise ValueError('Some action ID in action2nt mapping is out of range')
            if nonterm < 0 or nonterm >= num_nt:
                raise ValueError('Some nonterminal ID in action2nt mapping is out of range')
        if shift_action in action2nt:
            raise ValueError('SHIFT action cannot also be NT(X) action')
        non_reduce = {shift_action}
        non_reduce.update(action2nt)
        if len(non_reduce) + 1 != num_actions:
            raise ValueError('Cannot have more than one REDUCE action IDs')

        super().__init__()
        self.num_words = num_words
        self.num_pos = num_pos
        self.num_nt = num_nt
        self.num_actions = num_actions
        self.shift_action = shift_action
        self.action2nt = action2nt
        self.word_dim = word_dim
        self.pos_dim = pos_dim
        self.nt_dim = nt_dim
        self.action_dim = action_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Parser states
        self._stack = []  # type: List[StackElement]
        self._buffer = []  # type: List[WordId]
        self._history = []  # type: List[ActionId]
        self._num_open_nt = 0
        self._started = False

        # Parser state encoders
        self.stack_lstm = StackLSTM(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.buffer_lstm = StackLSTM(  # can use LSTM, but this is easier
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.history_lstm = StackLSTM(  # can use LSTM, but this is more efficient
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

        # Composition
        self.compose_fwd_lstm = nn.LSTM(
            input_dim, input_dim, num_layers=num_layers, dropout=dropout)
        self.compose_bwd_lstm = nn.LSTM(
            input_dim, input_dim, num_layers=num_layers, dropout=dropout)

        # Transformations
        self.word2lstm = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(word_dim + pos_dim, input_dim)),
            ('relu', nn.ReLU())
        ]))
        self.nt2lstm = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(nt_dim, input_dim)),
            ('relu', nn.ReLU())
        ]))
        self.action2lstm = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(action_dim, input_dim)),
            ('relu', nn.ReLU())
        ]))
        self.fwdbwd2composed = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(2 * input_dim, input_dim)),
            ('relu', nn.ReLU())
        ]))
        self.lstms2summary = nn.Sequential(OrderedDict([  # Stack LSTMs to parser summary
            ('dropout', nn.Dropout(dropout)),
            ('linear', nn.Linear(3 * hidden_dim, hidden_dim)),
            ('relu', nn.ReLU())
        ]))
        self.summary2actions = nn.Linear(hidden_dim, num_actions)

        # Embeddings
        self.word_embs = nn.Embedding(num_words, word_dim)
        self.pos_embs = nn.Embedding(num_pos, pos_dim)
        self.nt_embs = nn.Embedding(num_nt, nt_dim)
        self.action_embs = nn.Embedding(num_actions, action_dim)

        # Guard parameters for stack, buffer, and action history
        self.stack_guard = nn.Parameter(torch.Tensor(input_dim))
        self.buffer_guard = nn.Parameter(torch.Tensor(input_dim))
        self.history_guard = nn.Parameter(torch.Tensor(input_dim))

        # Final embeddings
        self._word_emb = {}  # type: Dict[WordId, Variable]
        self._nt_emb = {}  # type: Dict[NTId, Variable]
        self._action_emb = {}  # type: Dict[ActionId, Variable]

    @property
    def stack_buffer(self) -> List[Union[Tree, WordId]]:
        return [x.subtree for x in self._stack]

    @property
    def input_buffer(self) -> List[WordId]:
        return list(reversed(self._buffer))

    @property
    def action_history(self) -> List[ActionId]:
        return list(self._history)

    @property
    def num_open_nt(self) -> int:
        return self._num_open_nt

    @property
    def finished(self) -> bool:
        return (len(self._stack) == 1
                and not self._stack[0].is_open_nt
                and len(self._buffer) == 0)

    def start(self, tagged_words: Sequence[Tuple[WordId, POSId]]) -> None:
        if len(tagged_words) == 0:
            raise ValueError('parser cannot be started with empty sequence of words')

        self._stack = []
        self._buffer = []
        self._history = []

        while len(self.stack_lstm) > 0:
            self.stack_lstm.pop()
        while len(self.buffer_lstm) > 0:
            self.buffer_lstm.pop()
        while len(self.history_lstm) > 0:
            self.history_lstm.pop()

        # Feed guards as inputs
        self.stack_lstm.push(self.stack_guard)
        self.buffer_lstm.push(self.buffer_guard)
        self.history_lstm.push(self.history_guard)

        # Initialize input buffer and its LSTM encoder
        words, pos_tags = tuple(zip(*tagged_words))
        self._prepare_embeddings(words, pos_tags)
        for word in reversed(words):
            self._buffer.append(word)
            assert word in self._word_emb
            self.buffer_lstm.push(self._word_emb[word])
        self._started = True

    def do_action(self, action: ActionId) -> None:
        if not self._started:
            raise RuntimeError('parser is not started yet, please call `start` method first')
        if action < 0 or action >= self.num_actions:
            raise ValueError('action ID is out of range')

        legal, message = self._is_legal(action)
        if not legal:
            raise RuntimeError(message)

        if action == self.shift_action:
            self._shift()
        elif action in self.action2nt:
            self._push_new_open_nt(self.action2nt[action])
        else:
            self._reduce()
        self._history.append(action)
        try:
            self.history_lstm.push(self._action_emb[action])
        except KeyError:
            raise KeyError('cannot find embedding for the action; '
                           'perhaps you forgot to call .start() beforehand?')

    def forward(self):
        if not self._started:
            raise RuntimeError('parser is not started yet, please call `start` method first')

        lstm_embs = [self.stack_lstm.top, self.buffer_lstm.top, self.history_lstm.top]
        assert all(emb is not None for emb in lstm_embs)
        lstms_emb = torch.cat(lstm_embs).view(1, -1)
        parser_summary = self.lstms2summary(lstms_emb)
        illegal_actions = self._get_illegal_actions()
        if illegal_actions.dim() == 0:
            illegal_actions = None
        return log_softmax(self.summary2actions(parser_summary),
                           restrictions=illegal_actions).view(-1)

    def _prepare_embeddings(self, words: Collection[WordId], pos_tags: Collection[POSId]):
        assert len(words) == len(pos_tags)
        nonterms = list(self.action2nt.values())
        actions = range(self.num_actions)

        volatile = not self.training
        word_indices = Variable(self._new(words).long().view(1, -1), volatile=volatile)
        pos_indices = Variable(self._new(pos_tags).long().view(1, -1), volatile=volatile)
        nt_indices = Variable(self._new(nonterms).long().view(1, -1), volatile=volatile)
        action_indices = Variable(self._new(actions).long().view(1, -1), volatile=volatile)

        word_embs = self.word_embs(word_indices).view(-1, self.word_dim)
        pos_embs = self.pos_embs(pos_indices).view(-1, self.pos_dim)
        nt_embs = self.nt_embs(nt_indices).view(-1, self.nt_dim)
        action_embs = self.action_embs(action_indices).view(-1, self.action_dim)

        final_word_embs = self.word2lstm(torch.cat([word_embs, pos_embs], dim=1))
        final_nt_embs = self.nt2lstm(nt_embs)
        final_action_embs = self.action2lstm(action_embs)

        self._word_emb = dict(zip(words, final_word_embs))
        self._nt_emb = dict(zip(nonterms, final_nt_embs))
        self._action_emb = dict(zip(actions, final_action_embs))

    def _shift(self) -> None:
        assert len(self._buffer) > 0
        assert len(self.buffer_lstm) > 0
        word = self._buffer.pop()
        self.buffer_lstm.pop()
        assert word in self._word_emb
        self._stack.append(StackElement(word, self._word_emb[word], False))
        self.stack_lstm.push(self._word_emb[word])

    def _reduce(self) -> None:
        children = []
        while len(self._stack) > 0 and not self._stack[-1].is_open_nt:
            children.append(self._stack.pop()[:-1])
        assert len(children) > 0
        assert len(self._stack) > 0

        children.reverse()
        child_subtrees, child_embs = zip(*children)
        open_nt = self._stack.pop()
        assert isinstance(open_nt.subtree, Tree)
        open_nt.subtree.extend(child_subtrees)
        composed_emb = self._compose(open_nt.emb, child_embs)
        self._stack.append(StackElement(open_nt.subtree, composed_emb, False))
        self._num_open_nt -= 1

    def _push_new_open_nt(self, nonterm: NTId) -> None:
        try:
            self._stack.append(
                StackElement(Tree(nonterm, []), self._nt_emb[nonterm], True))
            self.stack_lstm.push(self._nt_emb[nonterm])
        except KeyError:
            raise KeyError('cannot find embedding for the nonterminal; '
                           'perhaps you forgot to call .start() beforehand?')
        else:
            self._num_open_nt += 1

    def _compose(self, open_nt_emb: Variable, children_embs: Sequence[Variable]) -> Variable:
        assert open_nt_emb.dim() == 1
        assert all(x.dim() == 1 for x in children_embs)
        assert open_nt_emb.size(0) == self.input_dim
        assert all(x.size(0) == self.input_dim for x in children_embs)

        fwd_input = [open_nt_emb]
        bwd_input = [open_nt_emb]
        for i in range(len(children_embs)):
            fwd_input.append(children_embs[i])
            bwd_input.append(children_embs[-i - 1])

        fwd_input = torch.cat(fwd_input).view(-1, 1, self.input_dim)
        bwd_input = torch.cat(bwd_input).view(-1, 1, self.input_dim)
        fwd_output, _ = self.compose_fwd_lstm(fwd_input, self._init_compose_states())
        bwd_output, _ = self.compose_bwd_lstm(bwd_input, self._init_compose_states())
        fwd_emb = F.dropout(fwd_output[-1, 0], p=self.dropout, training=self.training)
        bwd_emb = F.dropout(bwd_output[-1, 0], p=self.dropout, training=self.training)
        return self.fwdbwd2composed(torch.cat([fwd_emb, bwd_emb]).view(1, -1)).view(-1)

    def _get_illegal_actions(self):
        illegal_actions = [action for action in range(self.num_actions)
                           if not self._is_legal(action)[0]]
        return self._new(illegal_actions).long()

    def _is_legal(self, action: ActionId) -> Tuple[bool, str]:
        if self.finished:
            return False, 'parsing algorithm already finished, cannot do more action'

        nt_actions = self.action2nt.keys()
        n = self.num_open_nt
        if action in nt_actions:  # NT(X)
            if len(self._buffer) == 0:
                return False, 'cannot do NT(X) when input buffer is empty'
            elif n >= self.MAX_OPEN_NT:
                return False, 'max number of open nonterminals is reached'
            else:
                return True, ''
        elif action == self.shift_action:  # SHIFT
            if len(self._buffer) == 0:
                return False, 'cannot SHIFT when input buffer is empty'
            elif n == 0:
                return False, 'cannot SHIFT when no open nonterminal exists'
            else:
                return True, ''
        else:  # REDUCE
            last_is_nt = len(self._history) > 0 and self._history[-1] in nt_actions
            if last_is_nt:
                return False, 'cannot REDUCE when top of stack is an open nonterminal'
            elif n < 2 and len(self._buffer) > 0:
                return False, 'cannot REDUCE because there are words not SHIFT-ed yet'
            else:
                return True, ''

    def reset_parameters(self) -> None:
        # Stack LSTMs
        for name in ['stack', 'buffer', 'history']:
            lstm = getattr(self, f'{name}_lstm')
            for pname, pval in lstm.named_parameters():
                if pname.startswith('lstm.weight'):
                    init.orthogonal(pval)
                else:
                    assert pname.startswith('lstm.bias') or pname in ('h0', 'c0')
                    init.constant(pval, 0.)

        # Composition
        for name in ['fwd', 'bwd']:
            lstm = getattr(self, f'compose_{name}_lstm')
            for pname, pval in lstm.named_parameters():
                if pname.startswith('weight'):
                    init.orthogonal(pval)
                else:
                    assert pname.startswith('bias')
                    init.constant(pval, 0.)

        # Transformations
        gain = init.calculate_gain('relu')
        for name in ['word', 'nt', 'action']:
            layer = getattr(self, f'{name}2lstm')
            init.xavier_uniform(layer.linear.weight, gain=gain)
            init.constant(layer.linear.bias, 1.)
        init.xavier_uniform(self.fwdbwd2composed.linear.weight, gain=gain)
        init.constant(self.fwdbwd2composed.linear.bias, 1.)
        init.xavier_uniform(self.lstms2summary.linear.weight, gain=gain)
        init.constant(self.lstms2summary.linear.bias, 1.)
        init.xavier_uniform(self.summary2actions.weight)
        init.constant(self.summary2actions.bias, 0.)

        # Embeddings
        for name in ['word', 'pos', 'nt', 'action']:
            layer = getattr(self, f'{name}_emb')
            init.uniform(layer.weight, -0.01, 0.01)

        # Guards
        for name in ['stack', 'buffer', 'history']:
            guard = getattr(self, f'{name}_guard')
            init.uniform(guard, -0.01, 0.01)

    def _init_compose_states(self) -> Tuple[Variable, Variable]:
        h0 = Variable(self._new(self.num_layers, 1, self.input_dim).zero_())
        c0 = Variable(self._new(self.num_layers, 1, self.input_dim).zero_())
        return (h0, c0)

    def _new(self, *args, **kwargs):
        return next(self.parameters()).data.new(*args, **kwargs)
