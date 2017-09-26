from rnng.oracle import DiscOracle, GenOracle


class TestDiscOracle:
    def test_from_string(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
        expected_actions = ['NT(S)', 'NT(NP)', 'SHIFT', 'REDUCE', 'NT(VP)', 'SHIFT', 'NT(NP)',
                            'SHIFT', 'REDUCE', 'REDUCE', 'REDUCE']
        expected_words = ['John', 'loves', 'Mary']
        expected_pos_tags = ['NNP', 'VBZ', 'NNP']

        oracle = DiscOracle.from_parsed_string(s)

        assert [str(a) for a in oracle.actions] == expected_actions
        assert oracle.words == expected_words
        assert oracle.pos_tags == expected_pos_tags


class TestGenOracle:
    def test_from_string(self):
        s = '(S (NP (NNP John)) (VP (VBZ loves) (NP (NNP Mary))))'
        expected_actions = ['NT(S)', 'NT(NP)', 'GEN(John)', 'REDUCE', 'NT(VP)', 'GEN(loves)',
                            'NT(NP)', 'GEN(Mary)', 'REDUCE', 'REDUCE', 'REDUCE']
        expected_words = ['John', 'loves', 'Mary']
        expected_pos_tags = ['NNP', 'VBZ', 'NNP']

        oracle = GenOracle.from_parsed_string(s)

        assert [str(a) for a in oracle.actions] == expected_actions
        assert oracle.words == expected_words
        assert oracle.pos_tags == expected_pos_tags
