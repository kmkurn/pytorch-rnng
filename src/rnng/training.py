from datetime import timedelta
import math
import sys
import time

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
          grad_clip: float = 5.) -> None:
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


def train_early_stopping(train_loader: DataLoader, dev_loader: DataLoader, parser: RNNGrammar,
                         optimizer: Optimizer, tol: float = 1e-4, patience: int = 20,
                         eval_interval: int = 1, on_ppl: bool = True, save_to: str = None,
                         **kwargs) -> None:
    if patience <= 0:
        raise ValueError('patience must be positive')
    if 'epoch_num' in kwargs:
        kwargs.pop('epoch_num')

    print('** Start training with early stopping', file=sys.stderr)
    start_time = time.time()
    counter = epoch_num = 0
    loss = evaluate(train_loader, parser)
    print(f'Epoch {epoch_num}:', end=' ', file=sys.stderr)
    print(f'nll-per-action {loss:.4f} | ppl-per-action {math.exp(loss):.4f}', file=sys.stderr)
    best_loss = evaluate(dev_loader, parser)
    print(f'** Evaluate on devset:', end=' ', file=sys.stderr)
    print(f'nll-per-action {best_loss:.4f} | ppl-per-action {math.exp(best_loss):.4f}',
          file=sys.stderr)
    best_loss = math.exp(best_loss) if on_ppl else best_loss
    while counter < patience:
        epoch_num += 1
        train(train_loader, parser, optimizer, epoch_num=epoch_num, **kwargs)
        if epoch_num % eval_interval == 0:
            loss = evaluate(dev_loader, parser)
            print(f'** Evaluate on devset:', end=' ', file=sys.stderr)
            print(f'nll-per-action {loss:.4f}', end=' | ', file=sys.stderr)
            print(f'ppl-per-action {math.exp(loss):.4f}', file=sys.stderr)
            loss = math.exp(loss) if on_ppl else loss
            if loss < best_loss - tol:
                print('** Found new best loss on devset', file=sys.stderr)
                if save_to is not None:
                    torch.save(parser.state_dict(), save_to)
                    print(f'** Model saved to {save_to}', file=sys.stderr)
                best_loss = loss
                counter = 0
            else:
                counter += 1
    train_runtime = timedelta(seconds=time.time()-start_time)
    print('** Training finished in', train_runtime, file=sys.stderr)
