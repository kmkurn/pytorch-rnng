import pytest
import torch
from torch.autograd import Variable

from rnng.decoding import greedy_decode
from rnng.models import DiscRNNGrammar


word2id = {'John': 0, 'loves': 1, 'Mary': 2}
pos2id = {'NNP': 0, 'VBZ': 1}
nt2id = {'S': 0, 'NP': 1, 'VP': 2}
action2id = {'NT(S)': 0, 'NT(NP)': 1, 'NT(VP)': 2, 'SHIFT': 3, 'REDUCE': 4}
nt2action = {0: 0, 1: 1, 2: 2}


@pytest.mark.skip(reason='API change')
def test_greedy_decode(mocker):
    words = [word2id[w] for w in ['John', 'loves', 'Mary']]
    pos_tags = [pos2id[p] for p in ['NNP', 'VBZ', 'NNP']]
    correct_actions = [
        action2id['NT(S)'],
        action2id['NT(NP)'],
        action2id['SHIFT'],
        action2id['REDUCE'],
        action2id['NT(VP)'],
        action2id['SHIFT'],
        action2id['NT(NP)'],
        action2id['SHIFT'],
        action2id['REDUCE'],
        action2id['REDUCE'],
        action2id['REDUCE']
    ]
    retvals = [Variable(
        torch.zeros(len(action2id)).scatter_(0, torch.LongTensor([a]), 1))
                for a in correct_actions]
    parser = DiscRNNGrammar(
        len(word2id), len(pos2id), len(nt2id), len(action2id),
        action2id['SHIFT'], nt2action)
    parser.start(list(zip(words, pos_tags)))
    mocker.patch.object(parser, 'forward', side_effect=retvals)

    result = greedy_decode(parser)

    assert len(result) == len(correct_actions)
    picked_actions, log_probs = tuple(zip(*result))
    assert list(picked_actions) == correct_actions


@pytest.mark.skip(reason='API change')
def test_greedy_decode_missing_action_nt(mocker):
    words = [word2id[w] for w in ['John', 'loves', 'Mary']]
    pos_tags = [pos2id[p] for p in ['NNP', 'VBZ', 'NNP']]
    correct_actions = [
        action2id['NT(S)'],
        action2id['NT(NP)'],
        action2id['SHIFT'],
        action2id['REDUCE'],
        action2id['NT(VP)'],
        action2id['SHIFT'],
        action2id['NT(NP)'],
        action2id['SHIFT'],
        action2id['REDUCE'],
        action2id['REDUCE'],
        action2id['REDUCE']
    ]
    retvals = [Variable(
        torch.zeros(len(action2id)).scatter_(0, torch.LongTensor([a]), 1))
                for a in correct_actions]
    parser = DiscRNNGrammar(
        len(word2id), len(pos2id), len(nt2id), len(action2id),
        action2id['SHIFT'], nt2action)
    parser.nt2action.popitem()
    parser.start(list(zip(words, pos_tags)))
    mocker.patch.object(parser, 'forward', side_effect=retvals)

    with pytest.raises(KeyError):
        greedy_decode(parser)
