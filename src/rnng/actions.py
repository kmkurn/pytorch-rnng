from rnng.typing import Action, NTLabel, Word


REDUCE: Action = 'REDUCE'
SHIFT: Action = 'SHIFT'


def NT(label: NTLabel) -> Action:
    return f'NT({label})'


def GEN(word: Word) -> Action:
    return f'GEN({word})'


def get_nonterm(action: Action) -> NTLabel:
    if action.startswith('NT(') and action.endswith(')'):
        return action[3:-1]
    raise ValueError(f'action {action} is not an NT action')


def get_word(action: Action) -> Word:
    if action.startswith('GEN(') and action.endswith(')'):
        return action[4:-1]
    raise ValueError(f'action {action} is not a GEN action')


def is_nt(action: Action) -> bool:
    return action.startswith('NT')


def is_gen(action: Action) -> bool:
    return action.startswith('GEN')
