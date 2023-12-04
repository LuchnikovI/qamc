#!/usr/bin/python3

# pylint: skip-file

import logging
import numpy as np
import h5py  # type: ignore
from hamiltonian_utils import get_random_isings_ensemble  # type: ignore
from transition_matrix import get_transition_matrix
from omegaconf import DictConfig, OmegaConf
import hydra

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run(cfg):
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    spins_number = [int(s) for s in cfg.parameters.spins_number]
    gamma = [float(g) for g in cfg.parameters.gamma]
    tau = [float(t) for t in cfg.parameters.tau]
    ensemble_size = int(cfg.parameters.ensemble_size)
    couplings_ensemble = str(cfg.couplings_ensemble.type)
    density = float(cfg.couplings_ensemble.density)
    couplings_ensemble_params = cfg.couplings_ensemble.params
    if couplings_ensemble_params is None:
        couplings_ensemble_params = {}
    else:
        couplings_ensemble_params = {
            str(k): float(v) for k, v in couplings_ensemble_params.items()
        }
    local_fields_ensemble = str(cfg.local_fields_ensemble.type)
    local_fields_ensemble_params = cfg.local_fields_ensemble.params
    if local_fields_ensemble_params is None:
        local_fields_ensemble_params = {}
    else:
        local_fields_ensemble_params = {
            str(k): float(v) for k, v in local_fields_ensemble_params.items()
        }
    result = h5py.File(f"{out_dir}/result.hdf5", "w")
    for sn in spins_number:
        for t in tau:
            for g in gamma:
                ensemble = get_random_isings_ensemble(
                    ensemble_size,
                    sn,
                    density,
                    couplings_ensemble,
                    couplings_ensemble_params,
                    local_fields_ensemble,
                    local_fields_ensemble_params,
                )
                transition = get_transition_matrix(ensemble, t, g)
                lmbd = np.linalg.eigvals(transition)
                ids = np.argsort(np.abs(lmbd), axis=1)
                lmbd = np.take_along_axis(lmbd, ids, 1)
                result.create_dataset(f"{sn}/{g}/{t}/relax", data=lmbd[:, -2])
                log.info("Finished: spins_number: %i, tau: %f, gamma: %f", sn, t, g)


if __name__ == "__main__":
    run()
