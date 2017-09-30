import abc
from typing import List, Sequence
from typing import Type  # noqa

from nltk.tree import Tree


NTLabel = str
Word = str


class Action:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def from_string(cls, line: str):
        pass


class ShiftAction(Action):
    def __str__(self) -> str:
        return 'SHIFT'

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_string(cls, line: str) -> 'ShiftAction':
        if line != 'SHIFT':
            raise ValueError('invalid string value for SHIFT action')
        else:
            return cls()


class ReduceAction(Action):
    def __str__(self) -> str:
        return 'REDUCE'

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_string(cls, line: str) -> 'ReduceAction':
        if line != 'REDUCE':
            raise ValueError('invalid string value for REDUCE action')
        else:
            return cls()


class NTAction(Action):
    def __init__(self, label: NTLabel) -> None:
        self.label = label

    def __str__(self) -> str:
        return f'NT({self.label})'

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_string(cls, line: str) -> 'NTAction':
        if not line.startswith('NT(') or not line.endswith(')'):
            raise ValueError('invalid string value for NT(X) action')
        else:
            start = line.find('(') + 1
            return cls(line[start:-1])


class GenAction(Action):
    def __init__(self, word: Word) -> None:
        self.word = word

    def __str__(self) -> str:
        return f'GEN({self.word})'

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_string(cls, line: str) -> 'GenAction':
        if not line.startswith('GEN(') or not line.endswith(')'):
            raise ValueError('invalid string value for GEN(w) action')
        else:
            start = line.find('(') + 1
            return cls(line[start:-1])


class Oracle:
    def __init__(self, actions: Sequence[Action], pos_tags: Sequence[NTLabel],
                 words: Sequence[Word]) -> None:
        self.actions = actions
        self.pos_tags = pos_tags
        self.words = words

    def __str__(self) -> str:
        raise NotImplementedError

    @classmethod
    def from_parsed_sent(cls, parsed_sent: Tree) -> 'Oracle':
        raise NotImplementedError

    @classmethod
    def from_string(cls, line: str) -> 'Oracle':
        raise NotImplementedError

    @classmethod
    def get_action_on_pos_node(cls, pos_node: Tree) -> Action:
        raise NotImplementedError

    @classmethod
    def get_actions(cls, tree: Tree) -> List[Action]:
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return [cls.get_action_on_pos_node(tree)]

        actions: List[Action] = [NTAction(tree.label())]
        for child in tree:
            actions.extend(cls.get_actions(child))
        actions.append(ReduceAction())
        return actions


class DiscOracle(Oracle):
    def __init__(self, actions: Sequence[Action], pos_tags: Sequence[NTLabel],
                 words: Sequence[Word]) -> None:
        shift_cnt = sum(1 if isinstance(a, ShiftAction) else 0 for a in actions)
        if len(words) != shift_cnt:
            raise ValueError('number of words should match number of SHIFT actions')
        if len(pos_tags) != len(words):
            raise ValueError('number of POS tags should match number of words')

        super().__init__(actions, pos_tags, words)

    def __str__(self) -> str:
        out = [' '.join(self.words), ' '.join(self.pos_tags)]
        out.extend([str(a) for a in self.actions])
        return '\n'.join(out)

    @classmethod
    def from_parsed_sent(cls, parsed_sent: Tree) -> 'DiscOracle':
        actions = cls.get_actions(parsed_sent)
        words, pos_tags = zip(*parsed_sent.pos())
        return cls(actions, list(pos_tags), list(words))

    @classmethod
    def get_action_on_pos_node(cls, pos_node: Tree) -> ShiftAction:
        if len(pos_node) != 1 or isinstance(pos_node[0], Tree):
            raise ValueError('input is not a valid POS node')
        return ShiftAction()

    @classmethod
    def from_string(cls, line: str) -> 'DiscOracle':
        rows = line.split('\n')
        if len(rows) < 3:
            raise ValueError('string must have at least 3 lines (words, POS tags, actions)')

        words = rows[0].strip().split()
        pos_tags = rows[1].strip().split()
        actions = [cls.get_action_from_string(a_str) for a_str in rows[2:]]
        return cls(actions, pos_tags, words)

    @staticmethod
    def get_action_from_string(line: str) -> Action:
        classes = [NTAction, ShiftAction, ReduceAction]  # type: List[Type[Action]]
        for cls in classes:
            try:
                return cls.from_string(line)
            except ValueError:
                continue
        else:
            raise ValueError(
                f"'{line}' is not a valid string for any discriminative parser action")


class GenOracle(Oracle):
    def __init__(self, actions: Sequence[Action], pos_tags: Sequence[NTLabel]) -> None:
        gen_cnt = sum(1 if isinstance(a, GenAction) else 0 for a in actions)
        if len(pos_tags) != gen_cnt:
            raise ValueError('number of POS tags should match number of GEN actions')

        words = [a.word for a in actions if isinstance(a, GenAction)]
        super().__init__(actions, pos_tags, words)

    def __str__(self) -> str:
        out = [' '.join(self.pos_tags)]
        out.extend([str(a) for a in self.actions])
        return '\n'.join(out)

    @classmethod
    def from_parsed_sent(cls, parsed_sent: Tree) -> 'GenOracle':
        actions = cls.get_actions(parsed_sent)
        _, pos_tags = zip(*parsed_sent.pos())
        return cls(actions, list(pos_tags))

    @classmethod
    def get_action_on_pos_node(cls, pos_node: Tree) -> GenAction:
        if len(pos_node) != 1 or isinstance(pos_node[0], Tree):
            raise ValueError('input is not a valid POS node')
        return GenAction(pos_node[0])

    @classmethod
    def from_string(cls, line: str) -> 'GenOracle':
        rows = line.split('\n')
        if len(rows) < 2:
            raise ValueError('string must have at least 2 lines (POS tags, actions)')

        pos_tags = rows[0].strip().split()
        actions = [cls.get_action_from_string(a_str) for a_str in rows[1:]]
        return cls(actions, pos_tags)

    @staticmethod
    def get_action_from_string(line: str) -> Action:
        classes = [NTAction, GenAction, ReduceAction]  # type: List[Type[Action]]
        for cls in classes:
            try:
                return cls.from_string(line)
            except ValueError:
                continue
        else:
            raise ValueError(
                f"'{line}' is not a valid string for any generative parser action")
