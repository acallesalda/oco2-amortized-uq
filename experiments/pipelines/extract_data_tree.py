"""extract_data_tree.py
Saving the simulated OCO-2 data as a datatree.
"""
import logging
logger = logging.getLogger(__name__)
import sys
import os
sys.path.append("/workspace/project/code")

from data.processing import get_oco2_data, filter_variable, add_variable, extract_variables, flatten_tree
from pathlib import Path
import hydra
import h5py 

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np



@hydra.main(version_base=None, config_path="conf", config_name="extract_data_tree")
def main(cfg):

    data_root = Path(cfg.data.soundings)
    dt = get_oco2_data(data_root)

    # adding split to data.
    np.random.seed(cfg.data.seed)
    dt = dt.map_over_datasets(
        add_variable, 
        "split", 
        ("sounding", ), 
        lambda x:np.random.choice(["train", "test"], size = x.sizes['sounding'], p=[.8,.2])
    )
    path = Path(cfg.data.tree)
    Path(path).mkdir(parents=True, exist_ok=True)
    dt.to_zarr(path / 'oco2', mode="w")
    

if __name__=="__main__":
    main()