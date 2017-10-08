import math
import sys
import time
from typing import List, Tuple

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from rnng.models import RNNGrammar
from rnng.oracle import Oracle
from rnng.utils import MeanAggregate


def run_batch(parser: RNNGrammar, oracle: Oracle) -> Variable:
    parser.start(list(zip(oracle.words, oracle.pos_tags)))
    log_probs = []
    for action in oracle.actions:
        log_probs.append(parser().view(1, -1))
        action.execute_on(parser)
    outputs = torch.cat(log_probs)
    action_ids = [parser.action_store[a] for a in oracle.actions]
    targets = Variable(outputs.data.new(action_ids).long(), volatile=not parser.training)
    return F.nll_loss(outputs, targets)


def train(loader: DataLoader, parser: RNNGrammar, optimizer: Optimizer,
          log_interval: int = 100, epoch_num: int = 1,
          grad_clip: float = 5.):
    parser.train()
    loss = MeanAggregate()
    runtime = MeanAggregate()
    ppl = MeanAggregate()
    speed = MeanAggregate()

    start_time = time.time()
    for k, oracle in enumerate(loader):
        batch_start_time = time.time()
        batch_loss = run_batch(parser, oracle)
        optimizer.zero_grad()
        batch_loss.backward()
        clip_grad_norm(parser.parameters(), grad_clip)
        optimizer.step()
        batch_runtime = time.time() - batch_start_time

        loss.update(batch_loss.data[0])
        runtime.update(batch_runtime)
        ppl.update(math.exp(batch_loss.data[0]))
        speed.update(loader.batch_size / batch_runtime)

        if (k + 1) % log_interval == 0:
            print(f'Epoch {epoch_num} [{k+1}/{len(loader)}]:', end=' ', file=sys.stderr)
            print(f'nll-per-action {loss.mean:.4f} | ppl-per-action {ppl.mean:.4f}', end=' | ',
                  file=sys.stderr)
            print(f'{runtime.mean*1000:.2f}ms | {speed.mean:.2f} samples/s', file=sys.stderr)
            loss.reset()
            runtime.reset()
            ppl.reset()
            speed.reset()
    epoch_runtime = time.time() - start_time
    print(f'Epoch {epoch_num} done in {epoch_runtime:.2f}s', file=sys.stderr)


def evaluate(loader: DataLoader, parser: RNNGrammar) -> float:
    parser.eval()
    loss = MeanAggregate()
    for oracle in loader:
        batch_loss = run_batch(parser, oracle)
        loss.update(batch_loss.data[0])
    return loss.mean
