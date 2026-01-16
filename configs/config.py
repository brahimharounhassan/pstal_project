from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    model_name: str
    n_epochs: int
    patience: int
    _batch_size: int = None
    _device: str = None

    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device):
        self._device = device

@dataclass
class DatasetConfig:
    full: str
    small: str
    train: str
    dev: str
    tiny: str
    test: str


@dataclass
class PathsConfig:
    root: str
    output: str
    log: str
    model: str
    checkpoint: str

@dataclass
class HPtuningConfig:
    n_trials: int
    n_epochs: int

@dataclass
class Config:
    model: ModelConfig
    paths: PathsConfig
    dataset: DatasetConfig
    hptuning: HPtuningConfig


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**config["model_config"]),
        paths=PathsConfig(**config["paths"]),
        dataset=DatasetConfig(**config["dataset"]),
        hptuning=HPtuningConfig(**config["tuning_config"])
    )

CONFIG_FILE = "configs/config.yml"

config = load_config(
    path=CONFIG_FILE
)

MODEL_NAME = config.model.model_name
PATIENCE = config.model.patience
DEVICE = config.model.device
BATCH_SIZE = config.model.batch_size
N_EPOCHS = config.model.n_epochs

N_TRIALS_TUNER = config.hptuning.n_trials
N_EPOCH_TUNER = config.hptuning.n_epochs

ROOT_PATH = Path('..').resolve() / Path(config.paths.root)
OUTPUT_PATH =  ROOT_PATH / Path(config.paths.output)
LOG_PATH = ROOT_PATH / Path(config.paths.log)
MODEL_PATH = ROOT_PATH / Path(config.paths.model)
CHECKPOINT_PATH = ROOT_PATH / Path(config.paths.checkpoint)


DATA_FULL = ROOT_PATH / Path(config.dataset.full)
DATA_SMALL = ROOT_PATH / Path(config.dataset.small)
DATA_TRAIN = ROOT_PATH / Path(config.dataset.train)
DATA_DEV = ROOT_PATH / Path(config.dataset.dev)
DATA_TEST = ROOT_PATH / Path(config.dataset.test)

TARGET_UPOS = {"NOUN", "PROPN", "NUM"}
SUPERSENSE_COLUMN = "frsemcor:noun"

import os
for path in [MODEL_PATH, CHECKPOINT_PATH, OUTPUT_PATH, LOG_PATH]:
    os.makedirs(path, exist_ok=True)