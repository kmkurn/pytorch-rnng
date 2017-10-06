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

from rnng.actions import Action, ShiftAction, ReduceAction, NTAction
from rnng.typing import Word, POSTag, NTLabel, WordId, POSId, NTId, ActionId


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


class IllegalActionError(Exception):
    pass


class RNNGrammar(nn.Module, metaclass=abc.ABCMeta):
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
    def shift(self) -> None:
        pass

    @abc.abstractmethod
    def reduce(self) -> None:
        pass

    @abc.abstractmethod
    def push_nt(self, nonterm: NTId) -> None:
        pass


class DiscRNNGrammar(RNNGrammar):
    MAX_OPEN_NT = 100

    def __init__(self, word2id: Mapping[Word, WordId], pos2id: Mapping[POSTag, POSId],
                 nt2id: Mapping[NTLabel, NTId], action2id: Mapping[Action, ActionId],
                 word_dim: int = 32, pos_dim: int = 12, nt_dim: int = 60, action_dim: int = 16,
                 input_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.) -> None:
        if ShiftAction() not in action2id:
            raise ValueError('SHIFT action ID must be specified')
        if ReduceAction() not in action2id:
            raise ValueError('REDUCE action ID must be specified')

        num_words = len(word2id)
        num_pos = len(pos2id)
        num_nt = len(nt2id)
        num_actions = len(action2id)

        for wid in word2id.values():
            if wid < 0 or wid >= num_words:
                raise ValueError(f'word ID of {wid} is out of range')
        for pid in pos2id.values():
            if pid < 0 or pid >= num_pos:
                raise ValueError(f'POS tag ID of {pid} is out of range')
        for nid in nt2id.values():
            if nid < 0 or nid >= num_nt:
                raise ValueError(f'Nonterminal ID of {nid} is out of range')
        for aid in action2id.values():
            if aid < 0 or aid >= num_actions:
                raise ValueError(f'Action ID of {aid} is out of range')

        super().__init__()
        self.word2id = word2id
        self.pos2id = pos2id
        self.nt2id = nt2id
        self.action2id = action2id
        self.num_words = num_words
        self.num_pos = num_pos
        self.num_nt = num_nt
        self.num_actions = num_actions
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
        self._nt_emb = {}  # type: Variable
        self._action_emb = {}  # type: Variable

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
    def finished(self) -> bool:
        return (len(self._stack) == 1
                and not self._stack[0].is_open_nt
                and len(self._buffer) == 0)

    @property
    def started(self) -> bool:
        return self._started

    def start(self, tagged_words: Sequence[Tuple[Word, POSTag]]) -> None:
        if len(tagged_words) == 0:
            raise ValueError('parser cannot be started with empty sequence of words')
        for word, pos in tagged_words:
            if word not in self.word2id:
                raise ValueError(f"unknown word '{word}' encountered")
            if pos not in self.pos2id:
                raise ValueError(f"unknown POS tag '{pos}' encountered")

        self._stack = []
        self._buffer = []
        self._history = []
        self._num_open_nt = 0
        self._started = False

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
            assert word in self.word2id
            wid = self.word2id[word]
            assert wid in self._word_emb
            self.buffer_lstm.push(self._word_emb[wid])
        self._started = True

    def push_nt(self, nonterm: NTLabel) -> None:
        if nonterm not in self.nt2id:
            raise KeyError(f"unknown nonterminal '{nonterm}' encountered")
        action = NTAction(nonterm)
        if action not in self.action2id:
            raise KeyError(f"unknown action '{action}' encountered")

        self._verify_nt()
        self._push_nt(nonterm)
        self._history.append(action)
        aid = self.action2id[action]
        assert isinstance(self._action_emb, Variable)
        assert 0 <= aid < self._action_emb.size(0)
        self.history_lstm.push(self._action_emb[aid])

    def shift(self) -> None:
        self._verify_shift()
        self._shift()
        action = ShiftAction()
        self._history.append(action)
        assert action in self.action2id
        aid = self.action2id[action]
        assert isinstance(self._action_emb, Variable)
        assert 0 <= aid < self._action_emb.size(0)
        self.history_lstm.push(self._action_emb[aid])

    def reduce(self) -> None:
        self._verify_reduce()
        self._reduce()
        action = ReduceAction()
        assert action in self.action2id
        self._history.append(action)
        aid = self.action2id[action]
        assert isinstance(self._action_emb, Variable)
        assert 0 <= aid < self._action_emb.size(0)
        self.history_lstm.push(self._action_emb[aid])

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

    def _prepare_embeddings(self, words: Collection[Word], pos_tags: Collection[POSTag]):
        assert len(words) == len(pos_tags)
        assert all(w in self.word2id for w in words)
        assert all(p in self.pos2id for p in pos_tags)

        word_ids = [self.word2id[w] for w in words]
        pos_ids = [self.pos2id[p] for p in pos_tags]
        assert all(0 <= wid < self.num_words for wid in word_ids)
        assert all(0 <= pid < self.num_pos for pid in pos_ids)
        nt_ids = list(range(self.num_nt))
        action_ids = list(range(self.num_actions))

        volatile = not self.training
        word_indices = Variable(self._new(word_ids).long().view(1, -1), volatile=volatile)
        pos_indices = Variable(self._new(pos_ids).long().view(1, -1), volatile=volatile)
        nt_indices = Variable(self._new(nt_ids).long().view(1, -1), volatile=volatile)
        action_indices = Variable(self._new(action_ids).long().view(1, -1), volatile=volatile)

        word_embs = self.word_embs(word_indices).view(-1, self.word_dim)
        pos_embs = self.pos_embs(pos_indices).view(-1, self.pos_dim)
        nt_embs = self.nt_embs(nt_indices).view(-1, self.nt_dim)
        action_embs = self.action_embs(action_indices).view(-1, self.action_dim)

        final_word_embs = self.word2lstm(torch.cat([word_embs, pos_embs], dim=1))
        final_nt_embs = self.nt2lstm(nt_embs)
        final_action_embs = self.action2lstm(action_embs)

        self._word_emb = dict(zip(word_ids, final_word_embs))
        self._nt_emb = final_nt_embs
        self._action_emb = final_action_embs

    def _shift(self) -> None:
        assert len(self._buffer) > 0
        assert len(self.buffer_lstm) > 0
        word = self._buffer.pop()
        self.buffer_lstm.pop()
        assert word in self.word2id
        wid = self.word2id[word]
        assert wid in self._word_emb
        self._stack.append(StackElement(word, self._word_emb[wid], False))
        self.stack_lstm.push(self._word_emb[wid])

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
        assert self._num_open_nt >= 0

    def _push_nt(self, nonterm: NTLabel) -> None:
        nid = self.nt2id[nonterm]
        assert isinstance(self._nt_emb, Variable)
        assert 0 <= nid < self._nt_emb.size(0)
        self._stack.append(
            StackElement(Tree(nonterm, []), self._nt_emb[nid], True))
        self.stack_lstm.push(self._nt_emb[nid])
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
                           if self._is_legal(action)]
        return self._new(illegal_actions).long()

    def _is_legal(self, action: ActionId) -> bool:
        assert 0 <= action < self.num_actions
        try:
            if action == self.shift_action:
                self._verify_shift()
            elif action == self.reduce_action:
                self._verify_reduce()
            else:
                self._verify_nt()
        except IllegalActionError:
            return False
        else:
            return True

    def _verify_action(self) -> None:
        if not self._started:
            raise RuntimeError('parser is not started yet, please call `start` method first')
        if self.finished:
            raise RuntimeError('cannot do more action when parser is finished')

    def _verify_nt(self) -> None:
        self._verify_action()
        if len(self._buffer) == 0:
            raise IllegalActionError('cannot do NT(X) when input buffer is empty')
        if self._num_open_nt >= self.MAX_OPEN_NT:
            raise IllegalActionError('max number of open nonterminals is reached')

    def _verify_shift(self) -> None:
        self._verify_action()
        if len(self._buffer) == 0:
            raise IllegalActionError('cannot SHIFT when input buffer is empty')
        if self._num_open_nt == 0:
            raise IllegalActionError('cannot SHIFT when no open nonterminal exists')

    def _verify_reduce(self) -> None:
        self._verify_action()
        last_is_nt = len(self._history) > 0 and isinstance(self._history[-1], NTAction)
        if last_is_nt:
            raise IllegalActionError(
                'cannot REDUCE when top of stack is an open nonterminal')
        if self._num_open_nt < 2 and len(self._buffer) > 0:
            raise IllegalActionError(
                'cannot REDUCE because there are words not SHIFT-ed yet')

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
