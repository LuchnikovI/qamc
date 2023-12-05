from dataclasses import dataclass
from typing import List
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore

@dataclass
class ParametersConfig:
    spins_number: List[int]
    gamma: List[float]
    tau: List[float]
    ensemble_size: int
    batch_size: int

@dataclass
class CouplingsEnsembleConfig:
    distribution_type: str
    density: float
    params: DictConfig

@dataclass
class LocalFieldsEnsembleConfig:
    distribution_type: str
    params: DictConfig

@dataclass
class Config:
    parameters: ParametersConfig
    couplings_ensemble: CouplingsEnsembleConfig
    local_fields_ensemble: LocalFieldsEnsembleConfig

cs = ConfigStore.instance()
cs.store(name="Config", node=Config)
    