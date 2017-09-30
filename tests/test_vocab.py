from nltk.tree import Tree

from rnng.oracle import DiscOracle, GenOracle
from rnng.vocab import Vocabulary


def test_load_disc_oracle():
    s1 = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
    s2 = '(S (NP (NNP Mary)) (VP (VBZ hates) (NP (NNP John))))'  # poor John
    oracle1 = DiscOracle.from_parsed_sent(Tree.fromstring(s1))
    oracle2 = DiscOracle.from_parsed_sent(Tree.fromstring(s2))
    vocab = Vocabulary()

    vocab.load_oracles([oracle1, oracle2])

    words = ['John', 'Mary']
    pos_tags = ['NNP', 'VBZ']
    nt_labels = ['S', 'NP', 'VP']
    actions_str = ['NT(S)', 'NT(NP)', 'NT(VP)', 'SHIFT', 'REDUCE']
    assert sorted(words) == sorted(vocab.words)
    assert sorted(pos_tags) == sorted(vocab.pos_tags)
    assert sorted(nt_labels) == sorted(vocab.nt_labels)
    assert sorted(actions_str) == sorted(vocab.actions_str)
    word_ids = [vocab.get_word_id(w) for w in words]
    assert len(word_ids) == len(set(word_ids))
    assert all(0 <= wid and wid < len(word_ids) for wid in word_ids)
    pos_ids = [vocab.get_pos_id(p) for p in pos_tags]
    assert len(pos_ids) == len(set(pos_ids))
    assert all(0 <= pid and pid < len(pos_ids) for pid in pos_ids)
    nt_ids = [vocab.get_nt_id(l) for l in nt_labels]
    assert len(nt_ids) == len(set(nt_ids))
    assert all(0 <= nid and nid < len(nt_ids) for nid in nt_ids)
    action_ids = [vocab.get_action_id(a_str) for a_str in actions_str]
    assert len(action_ids) == len(set(action_ids))
    assert all(0 <= aid and aid < len(action_ids) for aid in action_ids)


def test_load_gen_oracle():
    s1 = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
    s2 = '(S (NP (NNP Mary)) (VP (VBZ hates) (NP (NNP John))))'  # poor John
    oracle1 = GenOracle.from_parsed_sent(Tree.fromstring(s1))
    oracle2 = GenOracle.from_parsed_sent(Tree.fromstring(s2))
    vocab = Vocabulary()

    vocab.load_oracles([oracle1, oracle2])

    words = ['John', 'Mary']
    pos_tags = ['NNP', 'VBZ']
    nt_labels = ['S', 'NP', 'VP']
    actions_str = ['NT(S)', 'NT(NP)', 'NT(VP)', 'GEN(John)', 'GEN(Mary)',
                   'GEN(loves)', 'GEN(hates)', 'REDUCE']
    assert sorted(words) == sorted(vocab.words)
    assert sorted(pos_tags) == sorted(vocab.pos_tags)
    assert sorted(nt_labels) == sorted(vocab.nt_labels)
    assert sorted(actions_str) == sorted(vocab.actions_str)
    word_ids = [vocab.get_word_id(w) for w in words]
    assert len(word_ids) == len(set(word_ids))
    assert all(0 <= wid and wid < len(word_ids) for wid in word_ids)
    pos_ids = [vocab.get_pos_id(p) for p in pos_tags]
    assert len(pos_ids) == len(set(pos_ids))
    assert all(0 <= pid and pid < len(pos_ids) for pid in pos_ids)
    nt_ids = [vocab.get_nt_id(l) for l in nt_labels]
    assert len(nt_ids) == len(set(nt_ids))
    assert all(0 <= nid and nid < len(nt_ids) for nid in nt_ids)
    action_ids = [vocab.get_action_id(a_str) for a_str in actions_str]
    assert len(action_ids) == len(set(action_ids))
    assert all(0 <= aid and aid < len(action_ids) for aid in action_ids)
