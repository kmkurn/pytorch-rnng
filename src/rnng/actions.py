import abc

from rnng.typing import NTLabel, Word


class Action(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abc.abstractmethod
    def __hash__(self) -> int:
        pass

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def from_string(cls, line: str) -> 'Action':
        for subclass in cls.__subclasses__():
            try:
                return subclass.from_string(line)
            except ValueError:
                pass  # continue to next subclass
        else:
            raise ValueError(f'no action found from string {line}')


class ShiftAction(Action):
    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__)

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return 'SHIFT'

    @classmethod
    def from_string(cls, line: str) -> 'ShiftAction':
        if line != 'SHIFT':
            raise ValueError('invalid string value for SHIFT action')
        else:
            return cls()


class ReduceAction(Action):
    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__)

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return 'REDUCE'

    @classmethod
    def from_string(cls, line: str) -> 'ReduceAction':
        if line != 'REDUCE':
            raise ValueError('invalid string value for REDUCE action')
        else:
            return cls()


class NTAction(Action):
    def __init__(self, label: NTLabel) -> None:
        self.label = label

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.label == other.label  # type: ignore

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return f'NT({self.label})'

    @classmethod
    def from_string(cls, line: str) -> 'NTAction':
        if not line.startswith('NT(') or not line.endswith(')'):
            raise ValueError('invalid string value for NT(X) action')
        else:
            return cls(line[3:-1])


class GenAction(Action):
    def __init__(self, word: Word) -> None:
        self.word = word

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.word == other.word  # type: ignore

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return f'GEN({self.word})'

    @classmethod
    def from_string(cls, line: str) -> 'GenAction':
        if not line.startswith('GEN(') or not line.endswith(')'):
            raise ValueError('invalid string value for GEN(w) action')
        else:
            return cls(line[4:-1])
