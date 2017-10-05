from typing import List, Tuple

from torch.autograd import Variable

from rnng.models import DiscRNNGrammar
from rnng.typing import ActionId


def greedy_decode(parser: DiscRNNGrammar) -> List[Tuple[ActionId, Variable]]:
    action2nt = {a: n for n, a in parser.nt2action.items()}
    result = []
    while not parser.finished:
        log_probs = parser()
        best_logprob, best_action = log_probs.data.max(0)
        best_action = best_action[0]
        best_logprob = best_logprob[0]
        result.append((best_action, best_logprob))
        if best_action == parser.shift_action:
            parser.shift()
        elif best_action == parser.reduce_action:
            parser.reduce()
        else:
            try:
                parser.push_nt(action2nt[best_action])
            except KeyError:
                raise KeyError(
                    f'action ID should be of an NT(X) action but '
                    "cannot be found in the parser's mapping: {best_action}")
    return result
