from typing import Collection, Iterator, Mapping, Union

from nltk.tree import Tree
from rnng.typing import NTId, NTLabel, Word, WordId


class TermStore(Collection[str]):
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


class ParseTreeMapper:
    def __init__(self, word2id: Mapping[Word, WordId], nt2id: Mapping[NTLabel, NTId]) -> None:
        words, word_ids = tuple(zip(*word2id.items()))
        nt_labels, nt_ids = tuple(zip(*nt2id.items()))

        if len(set(word_ids)) != len(word_ids):
            raise ValueError('word IDs should be unique')
        if len(set(nt_ids)) != len(nt_ids):
            raise ValueError('nonterminal IDs should be unique')

        self._word2id = word2id
        self._nt2id = nt2id
        self._id2word = dict(zip(word_ids, words))
        self._id2nt = dict(zip(nt_ids, nt_labels))

    def __call__(self, parse_tree: Tree) -> Tree:
        return self._map(parse_tree)

    def _map(self, root: Union[Tree, WordId]) -> Tree:
        if isinstance(root, WordId):
            return self._id2word[root]
        else:
            children = [self._map(child) for child in root]
            return Tree(self._id2nt[root.label()], children)


class MeanAggregate:
    def __init__(self) -> None:
        self.reset()

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.

    def reset(self):
        self.total = 0.
        self.count = 0.

    def update(self, val: float, size: int = 1):
        self.total += val * size
        self.count += size
