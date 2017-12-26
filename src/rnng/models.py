from typing import List, NamedTuple, Optional, Sequence, Sized, Tuple, Union, cast
from typing import Dict  # noqa

from nltk.tree import Tree
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from rnng.typing import WordId, NTId, ActionId


class EmptyStackError(Exception):
    def __init__(self):
        super().__init__('stack is already empty')


class StackLSTM(nn.Module, Sized):
    BATCH_SIZE = 1
    SEQ_LEN = 1

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.,
                 lstm_class=None) -> None:
        if input_size <= 0:
            raise ValueError(f'nonpositive input size: {input_size}')
        if hidden_size <= 0:
            raise ValueError(f'nonpositive hidden size: {hidden_size}')
        if num_layers <= 0:
            raise ValueError(f'nonpositive number of layers: {num_layers}')
        if dropout < 0. or dropout >= 1.:
            raise ValueError(f'invalid dropout rate: {dropout}')

        if lstm_class is None:
            lstm_class = nn.LSTM

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = lstm_class(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.h0 = nn.Parameter(torch.Tensor(num_layers, self.BATCH_SIZE, hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(num_layers, self.BATCH_SIZE, hidden_size))
        init_states = (self.h0, self.c0)
        self._states_hist = [init_states]
        self._outputs_hist = []  # type: List[Variable]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.lstm.named_parameters():
            if name.startswith('weight'):
                init.orthogonal(param)
            else:
                assert name.startswith('bias')
                init.constant(param, 0.)
        init.constant(self.h0, 0.)
        init.constant(self.c0, 0.)

    def forward(self, inputs: Variable) -> Tuple[Variable, Variable]:
        if inputs.size() != (self.input_size,):
            raise ValueError(
                f'expected input to have size ({self.input_size},), got {tuple(inputs.size())}'
            )
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


def log_softmax(inputs: Variable, restrictions: Optional[torch.LongTensor] = None) -> Variable:
    if restrictions is None:
        return F.log_softmax(inputs)

    if restrictions.dim() != 1:
        raise ValueError(f'restrictions must have dimension of 1, got {restrictions.dim()}')

    addend = Variable(
        inputs.data.new(inputs.size()).zero_().index_fill_(
            inputs.dim() - 1, restrictions, -float('inf')))
    return F.log_softmax(inputs + addend)


class StackElement(NamedTuple):
    subtree: Union[WordId, Tree]
    emb: Variable
    is_open_nt: bool


class DiscRNNG(nn.Module):
    MAX_OPEN_NT = 100
    REDUCE_ID = 0
    SHIFT_ID = 1

    def __init__(self,
                 num_words: int,
                 num_pos: int,
                 num_nt: int,
                 word_embedding_size: int = 32,
                 pos_embedding_size: int = 12,
                 nt_embedding_size: int = 60,
                 action_embedding_size: int = 16,
                 input_size: int = 128,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.,
                 ) -> None:
        super().__init__()
        self.num_words = num_words
        self.num_pos = num_pos
        self.num_nt = num_nt
        self.word_embedding_size = word_embedding_size
        self.pos_embedding_size = pos_embedding_size
        self.nt_embedding_size = nt_embedding_size
        self.action_embedding_size = action_embedding_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Parser states
        self._stack = []  # type: List[StackElement]
        self._buffer = []  # type: List[WordId]
        self._history = []  # type: List[ActionId]
        self._num_open_nt = 0

        # Embeddings
        self.word_embedding = nn.Embedding(self.num_words, self.word_embedding_size)
        self.pos_embedding = nn.Embedding(self.num_pos, self.pos_embedding_size)
        self.nt_embedding = nn.Embedding(self.num_nt, self.nt_embedding_size)
        self.action_embedding = nn.Embedding(self.num_actions, self.action_embedding_size)

        # Parser state encoders
        self.stack_encoder = StackLSTM(
            self.input_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout
        )
        self.stack_guard = nn.Parameter(torch.Tensor(self.input_size))
        self.buffer_encoder = StackLSTM(
            self.input_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout
        )
        self.buffer_guard = nn.Parameter(torch.Tensor(self.input_size))
        self.history_encoder = StackLSTM(
            self.input_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout
        )
        self.history_guard = nn.Parameter(torch.Tensor(self.input_size))

        # Compositions
        self.fwd_composer = nn.LSTM(
            self.input_size, self.input_size, num_layers=self.num_layers, dropout=self.dropout
        )
        self.bwd_composer = nn.LSTM(
            self.input_size, self.input_size, num_layers=self.num_layers, dropout=self.dropout
        )

        # Transformations
        self.word2encoder = nn.Sequential(
            nn.Linear(self.word_embedding_size + self.pos_embedding_size, self.hidden_size),
            nn.ReLU(),
        )
        self.nt2encoder = nn.Sequential(
            nn.Linear(self.nt_embedding_size, self.hidden_size),
            nn.ReLU(),
        )
        self.action2encoder = nn.Sequential(
            nn.Linear(self.action_embedding_size, self.hidden_size),
            nn.ReLU(),
        )
        self.fwdbwd2composed = nn.Sequential(
            nn.Linear(2 * self.input_size, self.input_size),
            nn.ReLU(),
        )
        self.encoders2summary = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(3 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.summary2actionlogprobs = nn.Linear(self.hidden_size, self.num_actions)

        # Final embeddings
        self._word_emb = {}  # type: Dict[WordId, Variable]
        self._nt_emb = {}  # type: Dict[NTId, Variable]
        self._action_emb = {}  # type: Dict[ActionId, Variable]

        self.reset_parameters()

    @property
    def num_actions(self) -> int:
        return self.num_nt + 2

    @property
    def finished(self) -> bool:
        return len(self._stack) == 1 and not self._stack[0].is_open_nt \
            and len(self._buffer) == 0

    def reset_parameters(self) -> None:
        # Embeddings
        for name in 'word pos nt action'.split():
            embedding = getattr(self, f'{name}_embedding')
            embedding.reset_parameters()

        # Encoders
        for name in 'stack buffer history'.split():
            encoder = getattr(self, f'{name}_encoder')
            encoder.reset_parameters()

        # Compositions
        for name in 'fwd bwd'.split():
            lstm = getattr(self, f'{name}_composer')
            for pname, pval in lstm.named_parameters():
                if pname.startswith('weight'):
                    init.orthogonal(pval)
                else:
                    assert pname.startswith('bias')
                    init.constant(pval, 0.)

        # Transformations
        gain = init.calculate_gain('relu')
        for name in 'word nt action'.split():
            layer = getattr(self, f'{name}2encoder')
            init.xavier_uniform(layer[0].weight, gain=gain)
            init.constant(layer[0].bias, 1.)
        init.xavier_uniform(self.fwdbwd2composed[0].weight, gain=gain)
        init.constant(self.fwdbwd2composed[0].bias, 1.)
        init.xavier_uniform(self.encoders2summary[1].weight, gain=gain)
        init.constant(self.encoders2summary[1].bias, 1.)
        init.xavier_uniform(self.summary2actionlogprobs.weight)
        init.constant(self.summary2actionlogprobs.bias, 0.)

        # Guards
        for name in 'stack buffer history'.split():
            guard = getattr(self, f'{name}_guard')
            init.constant(guard, 0.)

    def forward(self,
                words: Variable,
                pos_tags: Variable,
                actions: Variable) -> Variable:
        if words.dim() != 1:
            raise ValueError(f'expected words to have dimension of 1, got {words.dim()}')
        if words.size() != pos_tags.size():
            raise ValueError('expected POS tags to have size equal to words')
        if actions is not None and actions.dim() != 1:
            raise ValueError(f'expected actions to have dimension of 1, got {actions.dim()}')

        self._start(words, pos_tags, actions=actions)
        llh = 0.
        for action in actions:
            log_probs = self._compute_action_log_probs()
            llh += log_probs[action]
            action_id = action.data[0]
            if action_id == self.SHIFT_ID:
                if self._check_shift():
                    self._shift()
                else:
                    break
            elif action_id == self.REDUCE_ID:
                if self._check_reduce():
                    self._reduce()
                else:
                    break
            else:
                if self._check_push_nt():
                    self._push_nt(self._get_nt(action_id))
                else:
                    break
            self._append_history(action_id)
        return llh

    def decode(self, words: Variable, pos_tags: Variable) -> Tuple[List[ActionId], Tree]:
        self._start(words, pos_tags)
        while not self.finished:
            log_probs = self._compute_action_log_probs()
            max_action_id = torch.max(log_probs, dim=0)[1].data[0]
            if max_action_id == self.SHIFT_ID:
                if self._check_shift():
                    self._shift()
                else:
                    raise RuntimeError('most probable action is an illegal one')
            elif max_action_id == self.REDUCE_ID:
                if self._check_reduce():
                    self._reduce()
                else:
                    raise RuntimeError('most probable action is an illegal one')
            else:
                if self._check_push_nt():
                    self._push_nt(self._get_nt(max_action_id))
                else:
                    raise RuntimeError('most probable action is an illegal one')
            self._append_history(max_action_id)
        return list(self._history), self._stack[0].subtree

    def _start(self,
               words: Variable,
               pos_tags: Variable,
               actions: Optional[Variable] = None) -> None:
        # words: (seq_length,)
        # pos_tags: (seq_length,)
        # actions: (action_seq_length,)

        assert words.dim() == 1
        assert words.size() == pos_tags.size()
        if actions is not None:
            assert actions.dim() == 1

        self._stack = []
        self._buffer = []
        self._history = []
        self._num_open_nt = 0

        while len(self.stack_encoder) > 0:
            self.stack_encoder.pop()
        while len(self.buffer_encoder) > 0:
            self.buffer_encoder.pop()
        while len(self.history_encoder) > 0:
            self.history_encoder.pop()

        # Feed guards as inputs
        self.stack_encoder.push(self.stack_guard)
        self.buffer_encoder.push(self.buffer_guard)
        self.history_encoder.push(self.history_guard)

        # Initialize input buffer and its LSTM encoder
        self._prepare_embeddings(words, pos_tags, actions=actions)
        for word_id in reversed(words.data.tolist()):
            self._buffer.append(word_id)
            assert word_id in self._word_emb
            self.buffer_encoder.push(self._word_emb[word_id])

    def _prepare_embeddings(self,
                            words: Variable,
                            pos_tags: Variable,
                            actions: Optional[Variable] = None) -> None:
        # words: (seq_length,)
        # pos_tags: (seq_length,)
        # actions: (action_seq_length,)

        assert words.dim() == 1
        assert words.size() == pos_tags.size()
        if actions is not None:
            assert actions.dim() == 1

        if actions is None:
            actions = Variable(
                self._new(range(self.num_actions)), volatile=not self.training).long()
        nonterms = Variable(
            self._new(range(self.num_nt)), volatile=not self.training).long()

        word_embs = self.word_embedding(
            words.view(1, -1)).view(-1, self.word_embedding_size)
        pos_embs = self.pos_embedding(
            pos_tags.view(1, -1)).view(-1, self.pos_embedding_size)
        nt_embs = self.nt_embedding(
            nonterms.view(1, -1)).view(-1, self.nt_embedding_size)
        action_embs = self.action_embedding(
            actions.view(1, -1)).view(-1, self.action_embedding_size)

        final_word_embs = self.word2encoder(torch.cat([word_embs, pos_embs], dim=1))
        final_nt_embs = self.nt2encoder(nt_embs)
        final_action_embs = self.action2encoder(action_embs)

        self._word_emb = dict(zip(words.data.tolist(), final_word_embs))
        self._nt_emb = dict(zip(nonterms.data.tolist(), final_nt_embs))
        self._action_emb = dict(zip(actions.data.tolist(), final_action_embs))

    def _compute_action_log_probs(self) -> Variable:
        assert self.stack_encoder.top is not None
        assert self.buffer_encoder.top is not None
        assert self.history_encoder.top is not None

        concatenated = torch.cat([
            self.stack_encoder.top, self.buffer_encoder.top, self.history_encoder.top
        ]).view(1, -1)
        summary = self.encoders2summary(concatenated)
        illegal_actions = self._get_illegal_actions()
        return log_softmax(
            self.summary2actionlogprobs(summary),
            restrictions=illegal_actions
        ).view(-1)

    def _check_push_nt(self) -> bool:
        return len(self._buffer) > 0 and self._num_open_nt < self.MAX_OPEN_NT

    def _check_shift(self) -> bool:
        return len(self._buffer) > 0 and self._num_open_nt > 0

    def _check_reduce(self) -> bool:
        tos_is_open_nt = len(self._stack) > 0 and self._stack[-1].is_open_nt
        return self._num_open_nt > 0 and not tos_is_open_nt \
            and not (self._num_open_nt < 2 and len(self._buffer) > 0)

    def _append_history(self, action_id: ActionId) -> None:
        assert action_id in self._action_emb

        self._history.append(action_id)
        self.history_encoder.push(self._action_emb[action_id])

    def _push_nt(self, nt_id: NTId) -> None:
        assert self._check_push_nt()
        assert nt_id in self._nt_emb

        self._stack.append(
            StackElement(Tree(nt_id, []), self._nt_emb[nt_id], True))
        self.stack_encoder.push(self._nt_emb[nt_id])
        self._num_open_nt += 1

    def _shift(self) -> None:
        assert self._check_shift()
        assert len(self._buffer) > 0
        assert len(self.buffer_encoder) > 0
        assert self._buffer[-1] in self._word_emb

        word_id = self._buffer.pop()
        self.buffer_encoder.pop()
        self._stack.append(StackElement(word_id, self._word_emb[word_id], False))
        self.stack_encoder.push(self._word_emb[word_id])

    def _reduce(self) -> None:
        assert self._check_reduce()

        children = []
        while len(self._stack) > 0 and not self._stack[-1].is_open_nt:
            children.append(self._stack.pop()[:-1])
        assert len(children) > 0
        assert len(self._stack) > 0

        children.reverse()
        child_subtrees, child_embs = zip(*children)
        open_nt = self._stack.pop()
        assert isinstance(open_nt.subtree, Tree)
        parent_subtree = cast(Tree, open_nt.subtree)
        parent_subtree.extend(child_subtrees)
        composed_emb = self._compose(open_nt.emb, child_embs)
        self._stack.append(StackElement(parent_subtree, composed_emb, False))
        self._num_open_nt -= 1
        assert self._num_open_nt >= 0

    def _compose(self, open_nt_emb: Variable, children_embs: Sequence[Variable]) -> Variable:
        assert open_nt_emb.size() == (self.input_size,)
        assert all(x.size() == (self.input_size,) for x in children_embs)

        fwd_input = [open_nt_emb]
        bwd_input = [open_nt_emb]
        for i in range(len(children_embs)):
            fwd_input.append(children_embs[i])
            bwd_input.append(children_embs[-i - 1])

        # (n_children + 1, 1, input_size)
        fwd_input = torch.stack(fwd_input).unsqueeze(1)
        bwd_input = torch.stack(bwd_input).unsqueeze(1)
        # (n_children + 1, 1, input_size)
        fwd_output, _ = self.fwd_composer(fwd_input)
        bwd_output, _ = self.bwd_composer(bwd_input)
        # (input_size,)
        fwd_emb = F.dropout(fwd_output[-1, 0], p=self.dropout, training=self.training)
        bwd_emb = F.dropout(bwd_output[-1, 0], p=self.dropout, training=self.training)
        # (input_size,)
        return self.fwdbwd2composed(torch.cat([fwd_emb, bwd_emb]).view(1, -1)).view(-1)

    def _get_illegal_actions(self) -> Optional[torch.LongTensor]:
        illegal_action_ids = [
            action_id for action_id in range(self.num_actions) if not self._is_legal(action_id)
        ]
        if not illegal_action_ids:
            return None
        return self._new(illegal_action_ids).long()

    def _is_legal(self, action_id: int) -> bool:
        if action_id == self.SHIFT_ID:
            return self._check_shift()
        if action_id == self.REDUCE_ID:
            return self._check_reduce()
        return self._check_push_nt()

    def _get_nt(self, action_id: int) -> int:
        assert action_id >= 2
        return action_id - 2

    def _new(self, *args, **kwargs) -> torch.FloatTensor:
        return next(self.parameters()).data.new(*args, **kwargs)
