from typing import List, Sequence
import abc

from nltk.tree import Tree

from rnng.actions import GEN, NT, REDUCE, SHIFT, get_nonterm, get_word, is_gen, is_nt
from rnng.typing import Action, POSTag, Word


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

    def to_tree(self) -> Tree:
        stack = []
        pos_tags = list(reversed(self.pos_tags))
        words = list(reversed(self.words))
        for a in self.actions:
            if is_nt(a):
                stack.append(get_nonterm(a))
            elif a == REDUCE:
                children = []
                while stack and isinstance(stack[-1], Tree):
                    children.append(stack.pop())
                if not children or not stack:
                    raise ValueError(
                        f'invalid {REDUCE} action, please check if the actions are correct')
                parent = stack.pop()
                tree = Tree(parent, list(reversed(children)))
                stack.append(tree)
            else:
                tree = Tree(pos_tags.pop(), [words.pop()])
                stack.append(tree)
        if len(stack) != 1:
            raise ValueError('actions do not produce a single parse tree')
        return stack[0]

    @classmethod
    @abc.abstractmethod
    def from_tree(cls, tree: Tree):
        pass

    @classmethod
    @abc.abstractmethod
    def get_action_at_pos_node(cls, pos_node: Tree) -> Action:
        pass

    @classmethod
    def get_actions(cls, tree: Tree) -> List[Action]:
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return [cls.get_action_at_pos_node(tree)]

        actions: List[Action] = [NT(tree.label())]
        for child in tree:
            actions.extend(cls.get_actions(child))
        actions.append(REDUCE)
        return actions


class DiscOracle(Oracle):
    def __init__(self,
                 actions: Sequence[Action],
                 pos_tags: Sequence[POSTag],
                 words: Sequence[Word]) -> None:
        shift_cnt = sum(1 if a == SHIFT else 0 for a in actions)
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

    @classmethod
    def from_tree(cls, tree: Tree) -> 'DiscOracle':
        actions = cls.get_actions(tree)
        words, pos_tags = zip(*tree.pos())
        return cls(actions, list(pos_tags), list(words))

    @classmethod
    def get_action_at_pos_node(cls, pos_node: Tree) -> Action:
        if len(pos_node) != 1 or isinstance(pos_node[0], Tree):
            raise ValueError('input is not a valid POS node')
        return SHIFT


class GenOracle(Oracle):
    def __init__(self, actions: Sequence[Action], pos_tags: Sequence[POSTag]) -> None:
        gen_cnt = sum(1 if is_gen(a) else 0 for a in actions)
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
        return [get_word(a) for a in self.actions if is_gen(a)]

    @classmethod
    def from_tree(cls, tree: Tree) -> 'GenOracle':
        actions = cls.get_actions(tree)
        _, pos_tags = zip(*tree.pos())
        return cls(actions, list(pos_tags))

    @classmethod
    def get_action_at_pos_node(cls, pos_node: Tree) -> Action:
        if len(pos_node) != 1 or isinstance(pos_node[0], Tree):
            raise ValueError('input is not a valid POS node')
        return GEN(pos_node[0])
