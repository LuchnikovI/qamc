#!/usr/bin/python3

# pylint: skip-file

import logging
import numpy as np
import h5py  # type: ignore
from hamiltonian_utils import get_random_isings_ensemble  # type: ignore
from transition_matrix import get_transition_matrix
from omegaconf import DictConfig, OmegaConf
import hydra
from config import Config

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run(cfg: Config):
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    result = h5py.File(f"{out_dir}/result.hdf5", "w")
    for sn in cfg.parameters.spins_number:
        for t in cfg.parameters.tau:
            for g in cfg.parameters.gamma:
                ensemble = get_random_isings_ensemble(
                    cfg.parameters.batch_size,
                    sn,
                    cfg.couplings_ensemble.density,
                    cfg.couplings_ensemble.distribution_type,
                    {str(k): float(v) for k, v in cfg.couplings_ensemble.params.items()},
                    cfg.local_fields_ensemble.distribution_type,
                    {str(k): float(v) for k, v in cfg.local_fields_ensemble.params.items()},
                )
                assert cfg.parameters.ensemble_size % cfg.parameters.batch_size == 0, "Ensemble size must be multiple of batch size"
                batches_number = int(cfg.parameters.ensemble_size / cfg.parameters.batch_size)
                lmbd = np.zeros((0, 2 ** sn))
                for _ in range(batches_number):
                    transition = get_transition_matrix(ensemble, t, g)
                    lmbd = np.append(lmbd, np.linalg.eigvals(transition), axis=0)
                ids = np.argsort(np.abs(lmbd), axis=1)
                lmbd = np.take_along_axis(lmbd, ids, 1)
                result.create_dataset(f"{sn}/{g}/{t}/relax", data=lmbd[:, -2])
                log.info("Finished: spins_number: %i, tau: %f, gamma: %f", sn, t, g)


if __name__ == "__main__":
    run()
