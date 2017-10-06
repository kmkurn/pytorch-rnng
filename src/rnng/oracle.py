import abc
from typing import List, Sequence

from nltk.tree import Tree
from torch.utils.data import Dataset

from rnng.actions import Action, ShiftAction, ReduceAction, NTAction, GenAction
from rnng.typing import POSTag, Word
from rnng.utils import ItemStore


class Oracle(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def actions(self) -> List[Action]:
        pass

    @property
    @abc.abstractmethod
    def pos_tags(self) -> List[POSTag]:
        pass

    @property
    @abc.abstractmethod
    def words(self) -> List[Word]:
        pass

    @words.setter
    def words(self, new_words: List[Word]) -> None:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def from_parsed_sent(cls, parsed_sent: Tree):
        pass

    @classmethod
    @abc.abstractmethod
    def from_string(cls, line: str):
        pass

    @classmethod
    @abc.abstractmethod
    def get_action_at_pos_node(cls, pos_node: Tree) -> Action:
        pass

    @classmethod
    def get_actions(cls, tree: Tree) -> List[Action]:
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return [cls.get_action_at_pos_node(tree)]

        actions: List[Action] = [NTAction(tree.label())]
        for child in tree:
            actions.extend(cls.get_actions(child))
        actions.append(ReduceAction())
        return actions


class DiscOracle(Oracle):
    def __init__(self, actions: Sequence[Action], pos_tags: Sequence[POSTag],
                 words: Sequence[Word]) -> None:
        shift_cnt = sum(1 if isinstance(a, ShiftAction) else 0 for a in actions)
        if len(words) != shift_cnt:
            raise ValueError('number of words should match number of SHIFT actions')
        if len(pos_tags) != len(words):
            raise ValueError('number of POS tags should match number of words')

        self._actions = actions
        self._pos_tags = pos_tags
        self._words = words

    @property
    def actions(self) -> List[Action]:
        return list(self._actions)

    @property
    def pos_tags(self) -> List[POSTag]:
        return list(self._pos_tags)

    @property
    def words(self) -> List[Word]:
        return list(self._words)

    @words.setter
    def words(self, new_words: Sequence[Word]) -> None:
        if len(new_words) != len(self._words):
            raise ValueError('number of words should not change')
        self._words = new_words

    def __str__(self) -> str:
        out = [' '.join(self._words), ' '.join(self._pos_tags)]
        out.extend([str(a) for a in self._actions])
        return '\n'.join(out)

    @classmethod
    def from_parsed_sent(cls, parsed_sent: Tree) -> 'DiscOracle':
        actions = cls.get_actions(parsed_sent)
        words, pos_tags = zip(*parsed_sent.pos())
        return cls(actions, list(pos_tags), list(words))

    @classmethod
    def get_action_at_pos_node(cls, pos_node: Tree) -> Action:
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
    def __init__(self, actions: Sequence[Action], pos_tags: Sequence[POSTag]) -> None:
        gen_cnt = sum(1 if isinstance(a, GenAction) else 0 for a in actions)
        if len(pos_tags) != gen_cnt:
            raise ValueError('number of POS tags should match number of GEN actions')

        self._actions = actions
        self._pos_tags = pos_tags

    @property
    def actions(self) -> List[Action]:
        return list(self._actions)

    @property
    def pos_tags(self) -> List[POSTag]:
        return list(self._pos_tags)

    @property
    def words(self) -> List[Word]:
        return [a.word for a in self.actions if isinstance(a, GenAction)]

    @words.setter
    def words(self, new_words: Sequence[Word]) -> None:
        if len(new_words) != len(self.words):
            raise ValueError('number of words should not change')

        i = 0
        for word in new_words:
            while i < len(self._actions) and not isinstance(self._actions[i], GenAction):
                i += 1
            assert i < len(self._actions)
            self._actions[i].word = word  # type: ignore

    def __str__(self) -> str:
        out = [' '.join(self._pos_tags)]
        out.extend([str(a) for a in self._actions])
        return '\n'.join(out)

    @classmethod
    def from_parsed_sent(cls, parsed_sent: Tree) -> 'GenOracle':
        actions = cls.get_actions(parsed_sent)
        _, pos_tags = zip(*parsed_sent.pos())
        return cls(actions, list(pos_tags))

    @classmethod
    def get_action_at_pos_node(cls, pos_node: Tree) -> Action:
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


class OracleDataset(Dataset):
    def __init__(self, oracles: Sequence[Oracle]) -> None:
        self.oracles = oracles
        self.word2id = ItemStore()
        self.pos2id = ItemStore()
        self.nt2id = ItemStore()
        self.action2id = ItemStore()

        self.load()

    def load(self) -> None:
        for oracle in self.oracles:
            for word in oracle.words:
                self.word2id.add(word)
            for pos in oracle.pos_tags:
                self.pos2id.add(pos)
            for action in oracle.actions:
                self.action2id.add(action)
                if isinstance(action, NTAction):
                    self.nt2id.add(action.label)

    def __getitem__(self, index: int) -> Oracle:
        return self.oracles[index]

    def __len__(self) -> int:
        return len(self.oracles)
