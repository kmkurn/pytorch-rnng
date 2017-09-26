import abc
from typing import Iterable, List, Tuple, Union


NTLabel = str
Word = str


class ShiftAction:
    def __str__(self) -> str:
        return 'SHIFT'

    def __repr__(self) -> str:
        return str(self)


class ReduceAction:
    def __str__(self) -> str:
        return 'REDUCE'

    def __repr__(self) -> str:
        return str(self)


class NTAction:
    def __init__(self, label: NTLabel) -> None:
        self.label = label

    def __str__(self) -> str:
        return f'NT({self.label})'

    def __repr__(self) -> str:
        return str(self)


class GenAction:
    def __init__(self, word: Word) -> None:
        self.word = word

    def __str__(self) -> str:
        return f'GEN({self.word})'

    def __repr__(self) -> str:
        return str(self)


DiscParserAction = Union[ShiftAction, ReduceAction, NTAction]
GenParserAction = Union[GenAction, ReduceAction, NTAction]


class BaseOracle:
    __metaclass__ = abc.ABCMeta

    @classmethod
    def get_children_spans(cls, line: str) -> Iterable[Tuple[int, int]]:
        assert len(line) >= 2
        assert line[0] == '('
        assert line[-1] == ')'

        counter = 0
        for i, c in enumerate(line[1:-1], 1):
            if c == '(':
                counter += 1
                if counter == 1:
                    start = i
            elif c == ')':
                counter -= 1
                if counter == 0:
                    yield (start, i)

    @abc.abstractclassmethod
    def from_bracketed_string(cls, line: str):
        pass


class DiscOracle(BaseOracle):
    def __init__(self, actions: List[DiscParserAction], pos_tags: List[NTLabel],
                 words: List[Word]) -> None:
        shift_cnt = sum(1 if isinstance(a, ShiftAction) else 0 for a in actions)
        if len(words) != shift_cnt:
            raise ValueError('number of words should match number of SHIFT actions')
        if len(pos_tags) != len(words):
            raise ValueError('number of POS tags should match number of words')

        self.actions = actions
        self.words = words
        self.pos_tags = pos_tags

    @classmethod
    def from_bracketed_string(cls, line: str) -> 'DiscOracle':
        if len(line) < 2:
            raise ValueError('string must have length at least 2 (open and close brackets)')
        if line[0] != '(' or line[-1] != ')':
            raise ValueError(
                'string must begin and end with open and close bracket respectively')

        spans = list(cls.get_children_spans(line))
        if spans:
            nt_label = line[1:line.find(' ')]
            actions: List[DiscParserAction] = [NTAction(nt_label)]
            pos_tags, words = [], []
            for i, j in spans:
                child_oracle = cls.from_bracketed_string(line[i:j+1])
                actions.extend(child_oracle.actions)
                pos_tags.extend(child_oracle.pos_tags)
                words.extend(child_oracle.words)
            actions.append(ReduceAction())
            return cls(actions, pos_tags, words)
        else:
            pos_tag, word = line[1:-1].split()
            return cls([ShiftAction()], [pos_tag], [word])


class GenOracle(BaseOracle):
    def __init__(self, actions: List[GenParserAction], pos_tags: List[NTLabel]) -> None:
        gen_cnt = sum(1 if isinstance(a, GenAction) else 0 for a in actions)
        if len(pos_tags) != gen_cnt:
            raise ValueError('number of POS tags should match number of GEN actions')

        self.actions = actions
        self.pos_tags = pos_tags

    @classmethod
    def from_bracketed_string(cls, line: str) -> 'GenOracle':
        if len(line) < 2:
            raise ValueError('string must have length at least 2 (open and close brackets)')
        if line[0] != '(' or line[-1] != ')':
            raise ValueError(
                'string must begin and end with open and close bracket respectively')

        spans = list(cls.get_children_spans(line))
        if spans:
            nt_label = line[1:line.find(' ')]
            actions: List[GenParserAction] = [NTAction(nt_label)]
            pos_tags = []
            for i, j in spans:
                child_oracle = cls.from_bracketed_string(line[i:j+1])
                actions.extend(child_oracle.actions)
                pos_tags.extend(child_oracle.pos_tags)
            actions.append(ReduceAction())
            return cls(actions, pos_tags)
        else:
            pos_tag, word = line[1:-1].split()
            return cls([GenAction(word)], [pos_tag])

    @property
    def words(self) -> List[Word]:
        return [a.word for a in self.actions if isinstance(a, GenAction)]
