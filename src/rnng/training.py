import sys
import time
from typing import Tuple

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from rnng.models import RNNGrammar
from rnng.utils import MeanAggregate


def train(loader: DataLoader, parser: RNNGrammar, optimizer: Optimizer,
          log_interval: int = 100, epoch_num: int = 1,
          grad_clip: float = 5.) -> Tuple[float, float]:
    loss = MeanAggregate()
    runtime = MeanAggregate()
    ppl = MeanAggregate()
    speed = MeanAggregate()
    parser.train()

    for k, (words, pos_tags, actions) in enumerate(loader):
        batch_start_time = time.time()
        parser.start(zip(words, pos_tags))
        log_probs = []
        for action in actions:
            log_probs.append(parser().view(1, -1))
            parser.do_action(action)
        outputs = torch.cat(log_probs)
        targets = Variable(outputs.data.new(actions).long())
        batch_loss = F.nll_loss(outputs, targets)
        batch_ppl = batch_loss.exp()
        optimizer.zero_grad()
        batch_loss.backward()
        clip_grad_norm(parser.parameters(), grad_clip)
        optimizer.step()
        batch_runtime = time.time() - batch_start_time

        loss.update(batch_loss.data[0])
        runtime.update(batch_runtime)
        ppl.update(batch_ppl.data[0])
        speed.update(loader.batch_size / batch_runtime)

        if (k + 1) % log_interval == 0:
            print(f'Epoch {epoch_num} [{k+1}/{len(loader)}]:', end=' ', file=sys.stderr)
            print(f'action nll {loss.mean:.4f} | action ppl {ppl.mean:.4f}', end=' | ',
                  file=sys.stderr)
            print(f'{runtime.mean*1000:.2f}ms | {speed.mean:.2f} samples/s', file=sys.stderr)

    print(f'Epoch {epoch_num} done in {runtime.total:.2f}s', file=sys.stderr)
    return loss.mean, ppl.mean


def evaluate(loader: DataLoader, parser: RNNGrammar) -> Tuple[float, float]:
    parser.eval()
    loss = MeanAggregate()
    ppl = MeanAggregate()
    for words, pos_tags, actions in loader:
        parser.start(zip(words, pos_tags))
        log_probs = []
        for action in actions:
            log_probs.append(parser().view(1, -1))
            parser.do_action(action)
        outputs = torch.cat(log_probs)
        targets = Variable(outputs.data.new(actions).long(), volatile=True)
        batch_loss = F.nll_loss(outputs, targets)
        batch_ppl = batch_loss.exp()
        loss.update(batch_loss.data[0])
        ppl.update(batch_ppl.data[0])
    return loss.mean, ppl.mean