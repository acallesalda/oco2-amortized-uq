"""spectra.py
Plotting a single spectra.
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


from xarray.backends.api import open_datatree
@hydra.main(version_base=None, config_path="conf", config_name="figures")
def main(cfg):

    logger.info(f"Opening data ...")
    dt = open_datatree(
        Path(cfg.data.clean_tree),
        engine="zarr"
    )

    ds = dt['/Cluster08/ReferenceSounding6']


    wave_len = ds.wavelength.data
    radiance = ds.measured_radiance.data
    radiance = radiance[0,:]
    wave_len = wave_len[0,:]

    o2_band = wave_len < 0.8
    weak_co2_band = (wave_len > 0.8) * (wave_len < 1.8)
    strong_co2_band = wave_len > 1.8


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.vlines(wave_len[o2_band], 0, radiance[o2_band], linewidth=0.5)
    ax1.set_title("O$_2$ A-band")
    fig1.supxlabel("Wavelength ($\mu$m)")
    fig1.supylabel("Radiance")
    fig1.tight_layout()
    fig1.savefig("O2_A_band.pdf")


    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.vlines(wave_len[weak_co2_band], 0, radiance[weak_co2_band], linewidth=0.5)
    ax2.set_title("Weak CO$_2$")
    fig2.supxlabel("Wavelength ($\mu$m)")
    fig2.supylabel("Radiance")
    fig2.tight_layout()
    fig2.savefig("weak_CO_2_band.pdf")

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.vlines(wave_len[strong_co2_band], 0, radiance[strong_co2_band], linewidth=0.5)
    ax3.set_title("Strong CO$_2$")
    fig3.supxlabel("Wavelength ($\mu$m)")
    fig3.supylabel("Radiance")
    fig3.tight_layout()
    fig3.savefig("strong_CO_2_band.pdf")
    

if __name__=="__main__":
    main()