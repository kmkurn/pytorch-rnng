from collections import Counter
from typing import Collection, Iterator, List, Sequence

from rnng.oracle import Oracle, NTAction, Word, NTLabel


class TermCollection(Collection[str]):
    def __init__(self):
        self._term2id = {}
        self._id2term = {}

    def __contains__(self, term) -> bool:
        return term in self._term2id

    def __iter__(self) -> Iterator[str]:
        return iter(self._term2id)

    def __len__(self) -> int:
        return len(self._term2id)

    def add(self, term: str) -> None:
        if term not in self:
            size = len(self)
            self._term2id[term] = size
            self._id2term[size] = term
        assert len(self._term2id) == len(self._id2term)

    def get_id(self, term: str) -> int:
        return self._term2id[term]

    def get_term(self, ix: int) -> str:
        return self._id2term[ix]


class Vocabulary:
    def __init__(self, min_count: int = 2) -> None:
        self.min_count = min_count
        self._word_vocab = TermCollection()
        self._pos_vocab = TermCollection()
        self._nt_vocab = TermCollection()
        self._act_vocab = TermCollection()

    def load_oracles(self, oracles: Sequence[Oracle]) -> None:
        counter = Counter([w for oracle in oracles for w in oracle.words])
        for oracle in oracles:
            for word in oracle.words:
                if counter[word] >= self.min_count:
                    self._word_vocab.add(word)
            for pos in oracle.pos_tags:
                self._pos_vocab.add(pos)
            for act in oracle.actions:
                self._act_vocab.add(str(act))
                if isinstance(act, NTAction):
                    self._nt_vocab.add(act.label)

    @property
    def words(self) -> List[Word]:
        return list(self._word_vocab)

    @property
    def pos_tags(self) -> List[NTLabel]:
        return list(self._pos_vocab)

    @property
    def nt_labels(self) -> List[NTLabel]:
        return list(self._nt_vocab)

    @property
    def actions_str(self) -> List[str]:
        return list(self._act_vocab)

    def get_word_id(self, word: Word) -> int:
        return self._word_vocab.get_id(word)

    def get_pos_id(self, pos: NTLabel) -> int:
        return self._pos_vocab.get_id(pos)

    def get_nt_id(self, nt_label: NTLabel) -> int:
        return self._nt_vocab.get_id(nt_label)

    def get_action_id(self, action_str: str) -> int:
        return self._act_vocab.get_id(action_str)
