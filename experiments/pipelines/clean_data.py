"""clean_data.py
Clean the DataTree data and save it for a dataset.
"""
import logging
logger = logging.getLogger(__name__)
import sys
sys.path.append("/workspace/project/code")


import hydra
from pathlib import Path

import numpy as np
import pandas as pd
from xarray.backends.api import open_datatree

from data.processing import filter_variable, extract_variables, flatten_tree, \
get_missing, save_data, save_data_with_covariance,full_pipe


@hydra.main(version_base=None, config_path="conf", config_name="clean_data")
def main(cfg):

    logger.info(f"Opening data ...")
    path = Path(cfg.data.tree)
    dt = open_datatree(
        path / 'oco2', 
        engine="zarr"
    )
    logger.info(f"Data opened.")

    logger.info("Cleaning data ... ")
    variables = \
        list(cfg.target_vars) + \
        list(cfg.spectra_vars) + \
        list(cfg.aux_vars) + \
        ["split", "ID"]

    missing = dt.map_over_datasets(
        get_missing, 
        "measured_radiance"
    )
    missing = flatten_tree(missing)['no_measured_radiance']
    missing = np.all(missing, axis=0)

    dt = dt.map_over_datasets(
        full_pipe, 
        variables, 
        "true_xco2", 
        "wavelength_index", 
        missing
    )
    # Written under cfg.data.out_root, which may differ from the read-only
    # cfg.data.root the input tree was loaded from.
    clean_path = Path(cfg.data.clean_tree)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    dt.to_zarr(clean_path, mode="w")

    logger.info("Flatenning and storing...")
    path = Path(cfg.data.files)
    aux_vars = list(cfg.aux_vars) + list(cfg.target_vars) + ["ID"]
    aux_vars = [v for v in aux_vars if v != "retrieved_co2_profile_covariance"]
    data_vars = list(cfg.spectra_vars)
    for split in ["train", "test"]:
        split_dt = dt.map_over_datasets(filter_variable, "split", lambda z: z == split)
        save_data_with_covariance(
            split_dt, 
            path / split, 
            aux_vars, 
            data_vars,
            covariance_var="retrieved_co2_profile_covariance"  # Explicitly pass it
        )
    logger.info("Data stored.")


    
if __name__ == "__main__":
    main()