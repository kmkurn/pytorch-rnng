import json
import logging
import os
import random
import tarfile

from nltk.corpus.reader import BracketParseCorpusReader
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


class Trainer(object):
    def __init__(self,
                 train_corpus,
                 save_to,
                 dev_corpus=None,
                 encoding='utf-8',
                 rnng_type='discriminative',
                 lower=True,
                 min_freq=2,
                 word_embedding_size=32,
                 pos_embedding_size=12,
                 nt_embedding_size=60,
                 action_embedding_size=16,
                 input_size=128,
                 hidden_size=128,
                 num_layers=2,
                 dropout=0.5,
                 learning_rate=0.001,
                 max_epochs=20,
                 evalb=None,
                 evalb_params=None,
                 device=-1,
                 seed=25122017,
                 log_interval=10,
                 logger=None):
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

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

    def set_random_seed(self):
        self.logger.info('Setting random seed to %d', self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def prepare_for_serialization(self):
        self.logger.info('Preparing serialization directory in %s', self.save_to)
        os.makedirs(self.save_to, exist_ok=True)
        self.fields_dict_path = os.path.join(self.save_to, 'fields_dict.pkl')
        self.model_metadata_path = os.path.join(self.save_to, 'model_metadata.json')
        self.model_params_path = os.path.join(self.save_to, 'model_params.pth')
        self.artifacts_path = os.path.join(self.save_to, 'artifacts.tar.gz')

    def init_fields(self):
        self.WORDS = Field(pad_token=None, lower=self.lower)
        self.POS_TAGS = Field(pad_token=None)
        self.NONTERMS = Field(pad_token=None)
        self.ACTIONS = ActionField(self.NONTERMS)
        self.fields = [
            ('actions', self.ACTIONS), ('nonterms', self.NONTERMS),
            ('pos_tags', self.POS_TAGS), ('words', self.WORDS),
        ]

    def process_corpora(self):
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

    def build_vocabularies(self):
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

    def build_model(self):
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

    def build_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def run(self):
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

    def network(self, sample):
        words = sample.words.squeeze(1)
        pos_tags = sample.pos_tags.squeeze(1)
        actions = sample.actions.squeeze(1)
        llh = self.model(words, pos_tags, actions)
        return -llh, None

    def on_start(self, state):
        if state['train']:
            self.train_timer.reset()
        else:
            self.reset_meters()

    def on_start_epoch(self, state):
        self.reset_meters()
        self.epoch_timer.reset()

    def on_sample(self, state):
        self.batch_timer.reset()

    def on_forward(self, state):
        elapsed_time = self.batch_timer.value()
        self.loss_meter.add(state['loss'].data[0])
        self.speed_meter.add(state['sample'].words.size(1) / elapsed_time)
        if state['train'] and (state['t'] + 1) % self.log_interval == 0:
            epoch = (state['t'] + 1) / len(state['iterator'])
            loss, _ = self.loss_meter.value()
            speed, _ = self.speed_meter.value()
            self.logger.info(
                'Epoch %.4f (%.4fs): %.2f samples/sec | loss %.4f',
                epoch, elapsed_time, speed, loss)

    def on_end_epoch(self, state):
        epoch = state['epoch']
        elapsed_time = self.epoch_timer.value()
        loss, _ = self.loss_meter.value()
        speed, _ = self.speed_meter.value()
        self.logger.info('Epoch %d done (%.4fs): %.2f samples/sec | loss %.4f',
                         epoch, elapsed_time, speed, loss)
        self.save_model()
        if self.dev_iterator is not None:
            self.engine.test(self.network, self.dev_iterator)
            loss, _ = self.loss_meter.value()
            speed, _ = self.speed_meter.value()
            self.logger.info(
                'Evaluating on dev corpus: %.2f samples/sec | loss %.4f', speed, loss)

    def on_end(self, state):
        if state['train']:
            elapsed_time = self.train_timer.value()
            self.logger.info('Training done in %.4fs', elapsed_time)
            self.save_artifacts()

    def make_dataset(self, corpus):
        reader = BracketParseCorpusReader(
            *os.path.split(corpus), encoding=self.encoding, detect_blocks='sexpr')
        oracles = [DiscOracle.from_parsed_sent(s) for s in reader.parsed_sents()]
        examples = [make_example(x, self.fields) for x in oracles]
        return Dataset(examples, self.fields)

    def reset_meters(self):
        self.loss_meter.reset()
        self.speed_meter.reset()

    def save_artifacts(self):
        self.logger.info('Saving training artifacts to %s', self.artifacts_path)
        with tarfile.open(self.artifacts_path, 'w:gz') as f:
            artifact_names = 'fields_dict model_metadata model_params'.split()
            for name in artifact_names:
                path = getattr(self, f'{name}_path')
                f.add(path, arcname=os.path.basename(path))

    def save_model(self):
        self.logger.info('Saving model parameters to %s', self.model_params_path)
        torch.save(self.model.state_dict(), self.model_params_path)
