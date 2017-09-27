from torch.nn import Module, LSTMCell


class EmptyStackError(Exception):
    def __init__(self):
        super().__init__('stack is already empty')


class StackLSTM(Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._cell = LSTMCell(input_size, hidden_size)
        self._history = [('dummy_h0', 'dummy_c0')]

    def forward(self, inputs):
        next_hist = self._cell(inputs, self._history[-1])
        self._history.append(next_hist)
        return next_hist

    def pop(self):
        if len(self._history) > 1:
            return self._history.pop()
        else:
            raise EmptyStackError()

    @property
    def top(self):
        return self._history[-1][0] if len(self._history) > 1 else None
