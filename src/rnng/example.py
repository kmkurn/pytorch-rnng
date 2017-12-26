from typing import List, Tuple

from torchtext.data import Example, Field

from rnng.actions import is_nt, get_nonterm
from rnng.oracle import Oracle


def make_example(oracle: Oracle, fields: List[Tuple[str, Field]]):
    nonterms = [get_nonterm(a) for a in oracle.actions if is_nt(a)]
    return Example.fromlist(
        [oracle.actions, nonterms, oracle.pos_tags, oracle.words], fields
    )
