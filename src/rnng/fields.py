from collections import Counter

from torch.autograd import Variable
from torchtext.data import Field
from torchtext.vocab import Vocab

from rnng.actions import NTAction, ReduceAction, ShiftAction


class ActionField(Field):
    def __init__(self, nonterm_field: Field, **kwargs) -> None:
        unk_token = kwargs.pop('unk_token', None)
        pad_token = kwargs.pop('pad_token', None)
        super().__init__(unk_token=unk_token, pad_token=pad_token, **kwargs)
        self.nonterm_field = nonterm_field

    def build_vocab(self) -> None:
        specials = [str(ReduceAction()), str(ShiftAction())]
        for nonterm in self.nonterm_field.vocab.stoi:
            specials.append(str(NTAction(nonterm)))
        self.vocab = Vocab(Counter(), specials=specials)

    def numericalize(self, arr, **kwargs) -> Variable:
        arr = [[self._actionstr2id(s) for s in ex] for ex in arr]
        old_use_vocab = self.use_vocab  # type: ignore
        self.use_vocab = False
        arr = super().numericalize(arr, **kwargs)
        self.use_vocab = old_use_vocab
        return arr

    def _actionstr2id(self, s: str) -> int:
        if s in self.vocab.stoi:
            return self.vocab.stoi[s]
        # must be an unknown NT action, so we map it to NT(<unk>)
        action = NTAction(self.nonterm_field.unk_token)
        assert str(action) in self.vocab.stoi
        return self.vocab.stoi[str(action)]
