from torchtext.data import Dataset, Iterator


class SimpleIterator(Iterator):
    def __init__(self, dataset: Dataset, train: bool = True) -> None:
        super().__init__(dataset, 1, train=train, repeat=False, sort=False)

    def __iter__(self):
        return iter(self.data())
