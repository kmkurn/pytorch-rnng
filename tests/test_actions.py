import pytest

from rnng.actions import GEN, NT, REDUCE, SHIFT, get_nonterm, get_word, is_gen, is_nt


def test_reduce_action():
    assert REDUCE == 'REDUCE'


def test_shift_action():
    assert SHIFT == 'SHIFT'


def test_NT():
    assert NT('NP') == 'NT(NP)'


def test_GEN():
    assert GEN('John') == 'GEN(John)'


def test_get_nonterm():
    action = NT('NP')
    assert get_nonterm(action) == 'NP'


def test_get_nonterm_of_invalid_action():
    with pytest.raises(ValueError) as excinfo:
        get_nonterm(SHIFT)
    assert f'action {SHIFT} is not an NT action' in str(excinfo.value)


def test_get_word():
    action = GEN('John')
    assert get_word(action) == 'John'


def test_get_word_of_invalid_action():
    with pytest.raises(ValueError) as excinfo:
        get_word(SHIFT)
    assert f'action {SHIFT} is not a GEN action' in str(excinfo.value)


def test_is_nt():
    assert is_nt(NT('NP'))
    assert not is_nt(REDUCE)
    assert not is_nt(SHIFT)
    assert not is_nt(GEN('John'))


def test_is_gen():
    assert is_gen(GEN('John'))
    assert not is_gen(REDUCE)
    assert not is_gen(SHIFT)
    assert not is_gen(NT('NP'))
