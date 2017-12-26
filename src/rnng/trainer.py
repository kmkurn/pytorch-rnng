from typing import Optional, Tuple
import json
import logging
import os
import random
import re
import subprocess
import tarfile

from nltk.corpus.reader import BracketParseCorpusReader
from torch.autograd import Variable
from torchtext.data import Dataset, Field
import dill
import torch
import torch.optim as optim
import torchnet as tnt

from rnng.example import make_example
from rnng.fields import ActionField
from rnng.iterator import SimpleIterator
from rnng.models import DiscRNNG
from rnng.oracle import DiscOracle
from rnng.utils import add_dummy_pos, get_evalb_f1, id2parsetree


class Trainer(object):
    def __init__(self,
                 train_corpus: str,
                 save_to: str,
                 dev_corpus: Optional[str] = None,
                 encoding: str = 'utf-8',
                 rnng_type: str = 'discriminative',
                 lower: bool = True,
                 min_freq: int = 2,
                 word_embedding_size: int = 32,
                 pos_embedding_size: int = 12,
                 nt_embedding_size: int = 60,
                 action_embedding_size: int = 16,
                 input_size: int = 128,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 learning_rate: float = 0.001,
                 max_epochs: int = 20,
                 evalb: Optional[str] = None,
                 evalb_params: Optional[str] = None,
                 device: int = -1,
                 seed: int = 25122017,
                 log_interval: int = 10,
                 logger: Optional[logging.Logger] = None) -> None:
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        if evalb is None:
            evalb = 'evalb'

        self.train_corpus = train_corpus
        self.save_to = save_to
        self.dev_corpus = dev_corpus
        self.encoding = encoding
        self.rnng_type = 'discriminative'
        self.lower = lower
        self.min_freq = min_freq
        self.word_embedding_size = word_embedding_size
        self.pos_embedding_size = pos_embedding_size
        self.nt_embedding_size = nt_embedding_size
        self.action_embedding_size = action_embedding_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.evalb = evalb
        self.evalb_params = evalb_params
        self.device = device
        self.seed = seed
        self.log_interval = log_interval
        self.logger = logger

        self.loss_meter = tnt.meter.AverageValueMeter()
        self.speed_meter = tnt.meter.AverageValueMeter()
        self.batch_timer = tnt.meter.TimeMeter(None)
        self.epoch_timer = tnt.meter.TimeMeter(None)
        self.train_timer = tnt.meter.TimeMeter(None)
        self.engine = tnt.engine.Engine()
        self.ref_trees = []  # type: ignore
        self.hyp_trees = []  # type: ignore

    def set_random_seed(self) -> None:
        self.logger.info('Setting random seed to %d', self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def prepare_for_serialization(self) -> None:
        self.logger.info('Preparing serialization directory in %s', self.save_to)
        os.makedirs(self.save_to, exist_ok=True)
        self.fields_dict_path = os.path.join(self.save_to, 'fields_dict.pkl')
        self.model_metadata_path = os.path.join(self.save_to, 'model_metadata.json')
        self.model_params_path = os.path.join(self.save_to, 'model_params.pth')
        self.artifacts_path = os.path.join(self.save_to, 'artifacts.tar.gz')

    def init_fields(self) -> None:
        self.WORDS = Field(pad_token=None, lower=self.lower)
        self.POS_TAGS = Field(pad_token=None)
        self.NONTERMS = Field(pad_token=None)
        self.ACTIONS = ActionField(self.NONTERMS)
        self.fields = [
            ('actions', self.ACTIONS), ('nonterms', self.NONTERMS),
            ('pos_tags', self.POS_TAGS), ('words', self.WORDS),
        ]

    def process_corpora(self) -> None:
        self.logger.info('Reading train corpus from %s', self.train_corpus)
        self.train_dataset = self.make_dataset(self.train_corpus)
        self.train_iterator = SimpleIterator(self.train_dataset, device=self.device)
        self.dev_dataset = None
        self.dev_iterator = None
        if self.dev_corpus is not None:
            self.logger.info('Reading dev corpus from %s', self.dev_corpus)
            self.dev_dataset = self.make_dataset(self.dev_corpus)
            self.dev_iterator = SimpleIterator(
                self.dev_dataset, train=False, device=self.device)

    def build_vocabularies(self) -> None:
        self.logger.info('Building vocabularies')
        self.WORDS.build_vocab(self.train_dataset, min_freq=self.min_freq)
        self.POS_TAGS.build_vocab(self.train_dataset)
        self.NONTERMS.build_vocab(self.train_dataset)
        self.ACTIONS.build_vocab()

        self.num_words = len(self.WORDS.vocab)
        self.num_pos = len(self.POS_TAGS.vocab)
        self.num_nt = len(self.NONTERMS.vocab)
        self.num_actions = len(self.ACTIONS.vocab)
        self.logger.info(
            'Found %d words, %d POS tags, %d nonterminals, and %d actions',
            self.num_words, self.num_pos, self.num_nt, self.num_actions)

        self.logger.info('Saving fields dict to %s', self.fields_dict_path)
        torch.save(dict(self.fields), self.fields_dict_path, pickle_module=dill)

    def build_model(self) -> None:
        self.logger.info('Building model')
        model_args = (
            self.num_words, self.num_pos, self.num_nt)
        model_kwargs = dict(
            word_embedding_size=self.word_embedding_size,
            pos_embedding_size=self.pos_embedding_size,
            nt_embedding_size=self.nt_embedding_size,
            action_embedding_size=self.action_embedding_size,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        self.model = DiscRNNG(*model_args, **model_kwargs)
        if self.device >= 0:
            self.model.cuda(self.device)

        self.logger.info('Saving model metadata to %s', self.model_metadata_path)
        with open(self.model_metadata_path, 'w') as f:
            json.dump({'args': model_args, 'kwargs': model_kwargs}, f, sort_keys=True, indent=2)
        self.save_model()

    def build_optimizer(self) -> None:
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def run(self) -> None:
        self.set_random_seed()
        self.prepare_for_serialization()
        self.init_fields()
        self.process_corpora()
        self.build_vocabularies()
        self.build_model()
        self.build_optimizer()

        self.engine.hooks['on_start'] = self.on_start
        self.engine.hooks['on_start_epoch'] = self.on_start_epoch
        self.engine.hooks['on_sample'] = self.on_sample
        self.engine.hooks['on_forward'] = self.on_forward
        self.engine.hooks['on_end_epoch'] = self.on_end_epoch
        self.engine.hooks['on_end'] = self.on_end

        try:
            self.engine.train(
                self.network, self.train_iterator, self.max_epochs, self.optimizer)
        except KeyboardInterrupt:
            self.logger.info('Training interrupted, aborting')
            self.save_artifacts()

    def network(self, sample) -> Tuple[Variable, None]:
        words = sample.words.squeeze(1)
        pos_tags = sample.pos_tags.squeeze(1)
        actions = sample.actions.squeeze(1)
        llh = self.model(words, pos_tags, actions)
        training = self.model.training
        self.model.eval()
        _, hyp_tree = self.model.decode(words, pos_tags)
        self.model.train(training)
        hyp_tree = id2parsetree(
            hyp_tree, self.NONTERMS.vocab.itos, self.WORDS.vocab.itos)
        hyp_tree = add_dummy_pos(hyp_tree)
        self.hyp_trees.append(self.squeeze_whitespaces(str(hyp_tree)))
        return -llh, None

    def on_start(self, state: dict) -> None:
        if state['train']:
            self.train_timer.reset()
        else:
            self.reset_meters()
            self.model.eval()

    def on_start_epoch(self, state: dict) -> None:
        self.reset_meters()
        self.model.train()
        self.epoch_timer.reset()

    def on_sample(self, state: dict) -> None:
        self.batch_timer.reset()
        sample = state['sample']
        actions = [self.ACTIONS.vocab.itos[x] for x in sample.actions.squeeze(1).data]
        pos_tags = [self.POS_TAGS.vocab.itos[x] for x in sample.pos_tags.squeeze(1).data]
        words = [self.WORDS.vocab.itos[x] for x in sample.words.squeeze(1).data]
        tree = DiscOracle(actions, pos_tags, words).to_tree()
        self.ref_trees.append(self.squeeze_whitespaces(str(tree)))

    def on_forward(self, state: dict) -> None:
        elapsed_time = self.batch_timer.value()
        self.loss_meter.add(state['loss'].data[0])
        self.speed_meter.add(state['sample'].words.size(1) / elapsed_time)
        if state['train'] and (state['t'] + 1) % self.log_interval == 0:
            f1_score = self.compute_f1()
            epoch = (state['t'] + 1) / len(state['iterator'])
            loss, _ = self.loss_meter.value()
            speed, _ = self.speed_meter.value()
            self.logger.info(
                'Epoch %.4f (%.4fs): %.2f samples/sec | loss %.4f | F1 %.2f',
                epoch, elapsed_time, speed, loss, f1_score)

    def on_end_epoch(self, state: dict) -> None:
        iterator = SimpleIterator(self.train_dataset, train=False, device=self.device)
        self.engine.test(self.network, iterator)
        f1_score = self.compute_f1()
        epoch = state['epoch']
        elapsed_time = self.epoch_timer.value()
        loss, _ = self.loss_meter.value()
        speed, _ = self.speed_meter.value()
        self.logger.info('Epoch %d done (%.4fs): %.2f samples/sec | loss %.4f | F1 %.2f',
                         epoch, elapsed_time, speed, loss, f1_score)
        self.save_model()
        if self.dev_iterator is not None:
            self.engine.test(self.network, self.dev_iterator)
            f1_score = self.compute_f1()
            loss, _ = self.loss_meter.value()
            speed, _ = self.speed_meter.value()
            self.logger.info(
                'Evaluating on dev corpus: %.2f samples/sec | loss %.4f | F1 %.2f',
                speed, loss, f1_score)

    def on_end(self, state: dict) -> None:
        if state['train']:
            elapsed_time = self.train_timer.value()
            self.logger.info('Training done in %.4fs', elapsed_time)
            self.save_artifacts()

    def make_dataset(self, corpus: str) -> Dataset:
        reader = BracketParseCorpusReader(
            *os.path.split(corpus), encoding=self.encoding, detect_blocks='sexpr')
        oracles = [DiscOracle.from_tree(t) for t in reader.parsed_sents()]
        examples = [make_example(x, self.fields) for x in oracles]
        return Dataset(examples, self.fields)

    def reset_meters(self) -> None:
        self.loss_meter.reset()
        self.speed_meter.reset()
        self.ref_trees = []
        self.hyp_trees = []

    def save_artifacts(self) -> None:
        self.logger.info('Saving training artifacts to %s', self.artifacts_path)
        with tarfile.open(self.artifacts_path, 'w:gz') as f:
            artifact_names = 'fields_dict model_metadata model_params'.split()
            for name in artifact_names:
                path = getattr(self, f'{name}_path')
                f.add(path, arcname=os.path.basename(path))

    def save_model(self) -> None:
        self.logger.info('Saving model parameters to %s', self.model_params_path)
        torch.save(self.model.state_dict(), self.model_params_path)

    def compute_f1(self) -> float:
        ref_fname = os.path.join(self.save_to, 'reference.bracket')
        hyp_fname = os.path.join(self.save_to, 'hypothesis.bracket')
        with open(ref_fname, 'w') as ref_file, open(hyp_fname, 'w') as hyp_file:
            ref_file.write('\n'.join(self.ref_trees))
            hyp_file.write('\n'.join(self.hyp_trees))
        if self.evalb_params is None:
            args = [self.evalb, ref_file.name, hyp_file.name]
        else:
            args = [self.evalb, '-p', self.evalb_params, ref_fname, hyp_fname]
        res = subprocess.run(args, stdout=subprocess.PIPE, encoding='utf-8')
        return get_evalb_f1(res.stdout)

    @staticmethod
    def squeeze_whitespaces(s: str) -> str:
        return re.sub(r'(\n| )+', ' ', s)
