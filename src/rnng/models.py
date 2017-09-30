import math
from typing import Tuple
from typing import List  # noqa

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


class EmptyStackError(Exception):
    def __init__(self):
        super().__init__('stack is already empty')


class StackLSTM(nn.Module):
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
        self.h0 = nn.Parameter(torch.Tensor(num_layers, self.BATCH_SIZE, hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(num_layers, self.BATCH_SIZE, hidden_size))
        init_states = (self.h0, self.c0)
        self._states_hist = [init_states]
        self._outputs_hist = []  # type: List[Variable]
        self._lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, inputs: Variable) -> Tuple[Variable, Variable]:
        # inputs: input_size
        assert self._states_hist

        # Set seq_len and batch_size to 1
        inputs = inputs.view(self.SEQ_LEN, self.BATCH_SIZE, inputs.numel())
        next_outputs, next_states = self._lstm(inputs, self._states_hist[-1])
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

    def reset_parameters(self, init_states_std: float = .1) -> None:
        k = init_states_std * math.sqrt(3)
        init.uniform(self.h0, -k, k)
        init.uniform(self.c0, -k, k)

        for attr in dir(self._lstm):
            if attr.startswith('weight_'):
                init.orthogonal(getattr(self._lstm, attr))
            elif attr.startswith('bias_'):
                init.constant(getattr(self._lstm, attr), 0.)

    def __repr__(self) -> str:
        res = ('{}(input_size={input_size}, hidden_size={hidden_size}, '
               'num_layers={num_layers}, dropout={dropout})')
        return res.format(self.__class__.__name__, **self.__dict__)
