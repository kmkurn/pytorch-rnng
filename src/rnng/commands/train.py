import argparse
import json
import logging
import os
import random

from nltk.corpus.reader import BracketParseCorpusReader
from torchtext.data import Dataset, Field
import dill
import torch
import torch.optim as optim
import torchnet as tnt

from rnng.actions import Action, NTAction, ReduceAction, ShiftAction
from rnng.example import make_example
from rnng.iterator import SimpleIterator
from rnng.models import DiscRNNG
from rnng.oracle import DiscOracle


logging.basicConfig(format='%(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def make_parser(subparsers=None) -> argparse.ArgumentParser:
    description = 'Train RNNG on a given corpus.'
    if subparsers is None:
        parser = argparse.ArgumentParser(description=description)
    else:
        parser = subparsers.add_parser('train', description=description)

    parser.add_argument(
        '-t', '--train-corpus', required=True, metavar='FILE', help='path to train corpus')
    parser.add_argument(
        '-d', '--dev-corpus', metavar='FILE', help='path to dev corpus')
    parser.add_argument(
        '--encoding', default='utf-8', help='file encoding to use')
    parser.add_argument(
        '--rnng-type', choices='discriminative'.split(), metavar='TYPE',
        default='discriminative', help='type of RNNG to train')
    parser.add_argument(
        '--no-lower', action='store_false', dest='lower',
        help='whether not to lowercase the words')
    parser.add_argument(
        '--min-freq', type=int, default=2,
        help='minimum word frequency to be included in the vocabulary')
    parser.add_argument(
        '--word-embedding-size', type=int, default=32,
        help='dimension of word embeddings (default: 32)')
    parser.add_argument(
        '--pos-embedding-size', type=int, default=12,
        help='dimension of POS tag embeddings (default: 12)')
    parser.add_argument(
        '--nt-embedding-size', type=int, default=60,
        help='dimension of nonterminal embeddings (default: 12)')
    parser.add_argument(
        '--action-embedding-size', type=int, default=16,
        help='dimension of action embeddings (default: 16)')
    parser.add_argument(
        '--input-size', type=int, default=128,
        help='input dimension of the LSTM parser state encoders (default: 128)')
    parser.add_argument(
        '--hidden-size', type=int, default=128,
        help='hidden dimension of the LSTM parser state encoders (default: 128)')
    parser.add_argument(
        '--num-layers', type=int, default=2,
        help='number of layers of the LSTM parser state encoders and composers (default: 2)')
    parser.add_argument(
        '--dropout', type=float, default=0.5, help='dropout rate (default: 0.5)')
    parser.add_argument(
        '--learning-rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument(
        '--max-epochs', type=int, default=20,
        help='maximum number of epochs to train (default: 20)')
    parser.add_argument(
        '--patience', type=int, default=10,
        help='number of epochs to wait before early stopping (default: 10)')
    parser.add_argument(
        '-s', '--save-to', required=True, metavar='DIR',
        help='directory to save the training artifacts')
    parser.add_argument(
        '--evalb', help='path to evalb executable')
    parser.add_argument(
        '--evalb-params', help='path to evalb params file')
    parser.add_argument(
        '--log-interval', type=int, default=10,
        help='print logs every this number of iterations (default: 10)')
    parser.add_argument(
        '--seed', type=int, default=25122017, help='random seed')
    parser.add_argument(
        '--device', type=int, default=-1, help='GPU device to use (default: -1 for CPU)')
    parser.set_defaults(func=main)

    return parser


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_to, exist_ok=True)
    fields_dict_path = os.path.join(args.save_to, 'fields_dict.pkl')
    model_metadata_path = os.path.join(args.save_to, 'model_metadata.json')
    model_state_dict_path = os.path.join(args.save_to, 'model_state_dict.pth')

    def make_dataset(corpus, fields):
        reader = BracketParseCorpusReader(
            *os.path.split(corpus), encoding=args.encoding, detect_blocks='sexpr')
        oracles = [DiscOracle.from_parsed_sent(s) for s in reader.parsed_sents()]
        examples = [make_example(x, fields) for x in oracles]
        return Dataset(examples, fields)

    ACTIONS = Field(pad_token=None, unk_token=None)
    NONTERMS = Field(pad_token=None)
    POS_TAGS = Field(pad_token=None)
    WORDS = Field(pad_token=None, lower=args.lower)
    fields = [
        ('actions', ACTIONS), ('nonterms', NONTERMS), ('pos_tags', POS_TAGS), ('words', WORDS)]

    logger.info('Reading train corpus from %s', args.train_corpus)
    train_dataset = make_dataset(args.train_corpus, fields)
    train_iterator = SimpleIterator(train_dataset, device=args.device)
    dev_iterator = None
    if args.dev_corpus is not None:
        logger.info('Reading dev corpus from %s', args.dev_corpus)
        dev_dataset = make_dataset(args.dev_corpus, fields)
        dev_iterator = SimpleIterator(dev_dataset, train=False, device=args.device)

    logger.info('Building vocabulary')
    ACTIONS.build_vocab(train_dataset)
    NONTERMS.build_vocab(train_dataset)
    POS_TAGS.build_vocab(train_dataset)
    WORDS.build_vocab(train_dataset, min_freq=args.min_freq)

    num_words = len(WORDS.vocab)
    num_pos = len(POS_TAGS.vocab)
    num_nt = len(NONTERMS.vocab)
    num_actions = len(ACTIONS.vocab)
    logger.info(
        'Found %d words, %d POS tags, %d nonterminals, and %d actions',
        num_words, num_pos, num_nt, num_actions)

    logger.info('Saving fields dict to %s', fields_dict_path)
    torch.save(dict(fields), fields_dict_path, pickle_module=dill)

    logger.info('Creating models')
    action2nt = {}
    for action_id, actionstr in enumerate(ACTIONS.vocab.itos):
        action = Action.from_string(actionstr)
        if isinstance(action, NTAction):
            action2nt[action_id] = NONTERMS.vocab.stoi[action.label]
    model_args = (
        num_words, num_pos, num_nt, num_actions,
        ACTIONS.vocab.stoi[str(ShiftAction())], ACTIONS.vocab.stoi[str(ReduceAction())],
        action2nt)
    model_kwargs = dict(
        word_embedding_size=args.word_embedding_size,
        pos_embedding_size=args.pos_embedding_size,
        nt_embedding_size=args.nt_embedding_size,
        action_embedding_size=args.action_embedding_size,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    model = DiscRNNG(*model_args, **model_kwargs)

    logger.info('Saving model metadata to %s', model_metadata_path)
    with open(model_metadata_path, 'w') as f:
        json.dump({'args': model_args, 'kwargs': model_kwargs}, f, sort_keys=True, indent=2)

    loss_meter = tnt.meter.AverageValueMeter()
    speed_meter = tnt.meter.AverageValueMeter()
    batch_timer = tnt.meter.TimeMeter(None)
    epoch_timer = tnt.meter.TimeMeter(None)
    train_timer = tnt.meter.TimeMeter(None)
    engine = tnt.engine.Engine()
    best_loss = float('inf')
    current_patience = 0

    def net(sample):
        words = sample.words.squeeze(1)
        pos_tags = sample.pos_tags.squeeze(1)
        actions = sample.actions.squeeze(1)
        llh = model(words, pos_tags, actions)
        return -llh, None

    def reset_meters():
        loss_meter.reset()
        speed_meter.reset()

    def on_start(state):
        if state['train']:
            train_timer.reset()
        else:
            reset_meters()

    def on_start_epoch(state):
        reset_meters()
        epoch_timer.reset()

    def on_sample(state):
        batch_timer.reset()

    def on_forward(state):
        elapsed_time = batch_timer.value()
        loss_meter.add(state['loss'].data[0])
        speed_meter.add(state['sample'].words.size(1) / elapsed_time)
        if state['train'] and (state['t'] + 1) % args.log_interval == 0:
            epoch = (state['t'] + 1) / len(state['iterator'])
            loss, _ = loss_meter.value()
            speed, _ = speed_meter.value()
            logger.info(
                'Epoch %.4f (%.4fs): %.2f samples/sec | loss %.4f',
                epoch, elapsed_time, speed, loss)

    def on_end_epoch(state):
        epoch = state['epoch']
        elapsed_time = epoch_timer.value()
        loss, _ = loss_meter.value()
        speed, _ = speed_meter.value()
        logger.info('Epoch %d done (%.4fs): %.2f samples/sec | loss %.4f',
                    epoch, elapsed_time, speed, loss)
        if dev_iterator is None:
            logger.info('Saving model parameters to %s', model_state_dict_path)
            torch.save(model.state_dict(), model_state_dict_path)
        else:
            engine.test(net, dev_iterator)
            loss, _ = loss_meter.value()
            speed, _ = speed_meter.value()
            logger.info('Evaluating on dev corpus: %.2f samples/sec | loss %.4f', speed, loss)
            if loss < best_loss - 1e-6:
                logger.info('New best model found, saving to %s', model_state_dict_path)
                torch.save(model.state_dict(), model_state_dict_path)
                global current_patience
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= args.patience:
                    logger.info(
                        'No best model found after %d epochs, stopping', current_patience)
                    # Force training to stop
                    state['epoch'] = state['maxepoch'] + 1

    def on_end(state):
        if state['train']:
            elapsed_time = train_timer.value()
            logger.info('Training done in %.4fs', elapsed_time)

    engine.hooks['on_start'] = on_start
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_end'] = on_end

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    engine.train(net, train_iterator, args.max_epochs, optimizer)
