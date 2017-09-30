from rnng.utils import TermStore


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
