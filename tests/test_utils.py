from nltk.tree import Tree

from rnng.utils import TermStore, ParseTreeMapper


class TestTermStore:
    def test_contains(self):
        tc = TermStore()
        for c in list('aabbcc'):
            tc.add(c)

        assert 'a' in tc
        assert 'b' in tc
        assert 'c' in tc
        assert 'd' not in tc

    def test_iter(self):
        tc = TermStore()
        for c in list('aabbcc'):
            tc.add(c)

        assert sorted(list('abc')) == sorted(tc)

    def test_len(self):
        tc = TermStore()
        for c in list('aabbcc'):
            tc.add(c)

        assert len(tc) == 3

    def test_unique_ids(self):
        tc = TermStore()
        for c in list('aabbcc'):
            tc.add(c)

        ids = [tc.get_id(c) for c in list('abc')]
        assert len(ids) == len(set(ids))
        for i in ids:
            assert 0 <= i < len(tc)

    def test_invertible(self):
        tc = TermStore()
        for c in list('aabbcc'):
            tc.add(c)

        for c in list('abc'):
            assert tc.get_term(tc.get_id(c)) == c


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
