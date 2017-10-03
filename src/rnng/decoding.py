from typing import List, Tuple

from torch.autograd import Variable

from rnng.models import RNNGrammar
from rnng.typing import ActionId


def greedy_decode(parser: RNNGrammar) -> List[Tuple[ActionId, Variable]]:
    result = []
    while not parser.finished:
        log_probs = parser()
        best_logprob, best_action = log_probs.data.max(0)
        result.append((best_action[0], best_logprob[0]))
        parser.do_action(best_action[0])
    return result
