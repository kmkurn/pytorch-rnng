from torchtext.data import Field

from rnng.actions import NTAction, ReduceAction, ShiftAction
from rnng.fields import ActionField
from rnng.models import DiscRNNG


class TestActionField(object):
    def make_action_field(self):
        nonterm_field = Field(pad_token=None)
        return ActionField(nonterm_field)

    def test_init(self):
        nonterm_field = Field(pad_token=None)
        field = ActionField(nonterm_field)

        assert field.nonterm_field is nonterm_field
        assert field.unk_token is None
        assert field.pad_token is None

    def test_build_vocab(self):
        field = self.make_action_field()
        nonterms = 'S NP VP'.split()
        field.nonterm_field.build_vocab([nonterms])
        field.build_vocab()

        assert len(field.vocab) == len(field.nonterm_field.vocab) + 2
        assert field.vocab.stoi[str(ReduceAction())] == DiscRNNG.REDUCE_ID
        assert field.vocab.stoi[str(ShiftAction())] == DiscRNNG.SHIFT_ID
        for nonterm in nonterms:
            nid = field.nonterm_field.vocab.stoi[nonterm]
            action = NTAction(nonterm)
            assert field.vocab.stoi[str(action)] == nid + 2
        assert str(NTAction(field.nonterm_field.unk_token)) in field.vocab.stoi

    def test_numericalize(self):
        field = self.make_action_field()
        nonterms = 'S NP VP'.split()
        field.nonterm_field.build_vocab([nonterms])
        field.build_vocab()
        arr = [
            str(NTAction('S')),
            str(NTAction('NP')),
            str(NTAction('VP')),
            str(ShiftAction()),
            str(ReduceAction()),
        ]

        tensor = field.numericalize([arr], device=-1)

        assert tensor.size() == (len(arr), 1)
        assert tensor.squeeze().data.tolist() == [field.vocab.stoi[a] for a in arr]

    def test_numericalize_with_unknown_nt_action(self):
        field = self.make_action_field()
        nonterms = 'S NP VP'.split()
        field.nonterm_field.build_vocab([nonterms])
        field.build_vocab()
        arr = [
            str(NTAction('PP')),
        ]

        tensor = field.numericalize([arr], device=-1)

        assert tensor.squeeze().data.tolist() == [
            field.vocab.stoi[str(NTAction(field.nonterm_field.unk_token))]
        ]
