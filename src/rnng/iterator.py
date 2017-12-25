from typing import Optional

from torchtext.data import Dataset, Iterator


class SimpleIterator(Iterator):
    def __init__(self,
                 dataset: Dataset,
                 train: bool = True,
                 device: Optional[int] = None) -> None:
        super().__init__(dataset, 1, train=train, repeat=False, sort=False, device=device)
