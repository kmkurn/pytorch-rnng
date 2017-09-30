import os
from typing import Iterable

from nltk.corpus.reader import BracketParseCorpusReader
from nltk.tree import Tree


class Treebank:
    def __init__(self, corpus_file: str, lowercase: bool = True) -> None:
        self.corpus_file = corpus_file
        self.lowercase = lowercase
        self._reader = BracketParseCorpusReader(*os.path.split(corpus_file))

    def parsed_sents(self) -> Iterable[Tree]:
        if self.lowercase:
            return (self.lowercase_leaves(parsed_sent)
                    for parsed_sent in self._reader.parsed_sents())
        else:
            return self._reader.parsed_sents()

    @classmethod
    def lowercase_leaves(cls, tree):
        if isinstance(tree, str):
            return tree.lower()
        return Tree(tree.label(), [cls.lowercase_leaves(child) for child in tree])
