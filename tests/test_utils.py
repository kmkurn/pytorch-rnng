from nltk.tree import Tree

from rnng.utils import ItemStore, ParseTreeMapper, MeanAggregate


class TestItemStore:
    def test_contains(self):
        store = ItemStore()
        for c in list('aabbcc'):
            store.add(c)

        assert 'a' in store
        assert 'b' in store
        assert 'c' in store
        assert 'd' not in store

    def test_iter(self):
        store = ItemStore()
        for c in list('aabbcc'):
            store.add(c)

        assert sorted(list('abc')) == sorted(store)

    def test_len(self):
        store = ItemStore()
        for c in list('aabbcc'):
            store.add(c)

        assert len(store) == 3

    def test_unique_ids(self):
        store = ItemStore()
        for c in list('aabbcc'):
            store.add(c)

        ids = [store[c] for c in list('abc')]
        assert len(ids) == len(set(ids))
        for i in ids:
            assert 0 <= i < len(store)

    def test_invertible(self):
        store = ItemStore()
        for c in list('aabbcc'):
            store.add(c)

        for c in list('abc'):
            assert store.get_by_id(store[c]) == c


class TestParseTreeMapper:
    word2id = {'John': 0, 'loves': 1, 'Mary': 2}
    nt2id = {'S': 0, 'NP': 1, 'VP': 2}

    def test_call(self):
        parse_tree = Tree(self.nt2id['S'], [
            Tree(self.nt2id['NP'], [
                self.word2id['John']]),
            Tree(self.nt2id['VP'], [
                self.word2id['loves'],
                Tree(self.nt2id['NP'], [
                    self.word2id['Mary']
                ])
            ])
        ])
        exp_parse_tree = Tree('S', [
            Tree('NP', ['John']),
            Tree('VP', ['loves', Tree('NP', ['Mary'])])
        ])

        mapper = ParseTreeMapper(self.word2id, self.nt2id)

        assert str(mapper(parse_tree)) == str(exp_parse_tree)


class TestMeanAggregate:
    tol = 1e-4

    def is_close_to(self, x, y):
        return y - self.tol <= x and x <= y + self.tol

    def test_init(self):
        agg = MeanAggregate()

        assert self.is_close_to(agg.total, 0.)
        assert self.is_close_to(agg.count, 0.)
        assert self.is_close_to(agg.mean, 0.)

    def test_update(self):
        agg = MeanAggregate()
        agg.update(1.)
        agg.update(2.)
        agg.update(3.)

        assert self.is_close_to(agg.total, 6.)
        assert self.is_close_to(agg.count, 3.)
        assert self.is_close_to(agg.mean, 2.)
