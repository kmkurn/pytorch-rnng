from typing import Collection, Iterator


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
