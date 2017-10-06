from typing import List, Tuple

from torch.autograd import Variable

from rnng.actions import Action, ShiftAction, ReduceAction, NTAction, GenAction
from rnng.models import RNNGrammar


def greedy_decode(parser: RNNGrammar) -> List[Tuple[Action, Variable]]:
    result = []
    while not parser.finished:
        log_probs = parser()
        best_logprob, best_action_id = log_probs.data.max(0)
        best_action_id = best_action_id[0]
        best_logprob = best_logprob[0]
        best_action = parser.action_store.get_by_id(best_action_id)
        result.append((best_action, best_logprob))
        if isinstance(best_action, ShiftAction):
            parser.shift()
        elif isinstance(best_action, ReduceAction):
            parser.reduce()
        elif isinstance(best_action, NTAction):
            parser.push_nt(best_action.label)
        elif isinstance(best_action, GenAction):
            raise NotImplementedError('generative RNNG is not implemented yet')
        else:
            raise TypeError(f"unknown action type '{type(best_action)}'")
    return result
