import random

from torch.autograd import Variable
from torchtext.data import Dataset, Example, Field

from rnng.iterator import SimpleIterator


random.seed(12345)


class TestSimpleIterator(object):
    TEXT = Field()
    examples = [
        Example.fromlist(['John loves Mary'], [('text', TEXT)]),
        Example.fromlist(['Mary cries'], [('text', TEXT)]),
    ]
    dataset = Dataset(examples, [('text', TEXT)])
    TEXT.build_vocab(dataset)

    def make_iterator(self):
        return SimpleIterator(self.dataset, device=-1)

    def test_init_minimal(self):
        iterator = SimpleIterator(self.dataset)
        assert iterator.dataset is self.dataset
        assert iterator.batch_size == 1
        assert iterator.train
        assert iterator.device is None
        assert iterator.sort_key is None
        assert not iterator.sort
        assert not iterator.repeat
        assert iterator.shuffle == iterator.train
        assert not iterator.sort_within_batch

    def test_init_full(self):
        iterator = SimpleIterator(self.dataset, train=False, device=-1)
        assert not iterator.train
        assert iterator.device == -1

    def test_next(self):
        iterator = self.make_iterator()
        sample = next(iter(iterator))

        assert isinstance(sample.text, Variable)
        assert sample.text.size(1) == 1
