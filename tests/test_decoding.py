import torch
from torch.autograd import Variable
import pytest

from rnng.actions import ShiftAction, ReduceAction, NTAction
from rnng.decoding import greedy_decode
from rnng.models import DiscRNNG as DiscRNNGrammar
from rnng.utils import ItemStore


word2id = {'John': 0, 'loves': 1, 'Mary': 2}
pos2id = {'NNP': 0, 'VBZ': 1}
nt2id = {'S': 0, 'NP': 1, 'VP': 2}
actions = [NTAction('S'), NTAction('NP'), NTAction('VP'), ShiftAction(), ReduceAction()]
action_store = ItemStore()
for a in actions:
    action_store.add(a)


@pytest.mark.skip
def test_greedy_decode(mocker):
    words = ['John', 'loves', 'Mary']
    pos_tags = ['NNP', 'VBZ', 'NNP']
    correct_actions = [
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
    retvals = [Variable(
        torch.zeros(len(action_store)).scatter_(0, torch.LongTensor([action_store[a]]), 1))
                for a in correct_actions]
    parser = DiscRNNGrammar(word2id, pos2id, nt2id, action_store)
    parser.start(list(zip(words, pos_tags)))
    mocker.patch.object(parser, 'forward', side_effect=retvals)

    result = greedy_decode(parser)

    assert len(result) == len(correct_actions)
    picked_actions, log_probs = zip(*result)
    assert list(picked_actions) == correct_actions
