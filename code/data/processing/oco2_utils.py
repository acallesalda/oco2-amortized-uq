"""oco2.py
Tools for handling simulated OCO-2 data.
"""
import logging
logger = logging.getLogger(__name__)

import os
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import re  
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

o2_A_band = (0,1226)
weak_co2_band = (1226,2131)
strong_co2_band = (2131,-1)
META_COLUMNS=[
    "Footprint",
    "SlrZen",
    "Longitude",
    "Latitude",
    "AerType1",
    "AerType2",
    "Region"
]

def get_oco2_data(path:Path)-> xr.DataTree:
    """
    Load OCO-2 data from the specified path and organize it into an xarray DataTree structure.

    This function iterates through the directories (or clusters) within the provided path, 
    processes the data for each cluster, and updates a DataTree with the processed data. 
    The resulting DataTree contains a hierarchical representation of the dataset, 
    which can be used for further analysis or visualization.

    :param path: The root directory containing the OCO-2 data clusters.
    :type path: Path

    :return: A DataTree object containing the organized data from the specified directory.
    :rtype: xr.DataTree
    """
    data_dict = defaultdict(dict)
    dt = xr.DataTree(name="root", dataset = None)
    pbar = tqdm(path.iterdir(), file=open(os.devnull, 'w'))
    for cluster in pbar:
        pbar.set_description(f"Processing data in: {cluster.name}")
        logger.info(str(pbar))
        dt.update(get_cluster(cluster))

    return dt

def get_cluster(path:Path):
    """Gather all the data from a single cluster."""
    cluster_dt = xr.DataTree(name=path.name, dataset=None)
    meta_data = get_meta_data(path)
    i = 0
    datasets = []
    for ref_sounding in path.iterdir():
        if ref_sounding.suffix != '.csv':
            ds = get_repeats(ref_sounding, meta_data.iloc[i])
            ref_id = get_property(ref_sounding, "ref")
            cluster_dt[f"{path.name}/ReferenceSounding{ref_id}"] = xr.DataTree(dataset=ds)
            i += 1
    return cluster_dt

    

def get_repeats(path:Path, meta_data):
    """Geturn the sounding data stored at path."""
    group_data = {}
    groups = {
        "FilterVariables":fix_filter_variables,
        "RetrievalHeader":fix_retrieval_header, 
        "RetrievalResults":fix_retrieval_results, 
        "SpectralParameters":fix_spectral_parameters, 
        "StateVector":fix_state_vector
    }

    for group in groups.keys():
        group_data[group] = xr.load_dataset(
            path, engine='h5netcdf', group=group, phony_dims='access'
        )
    for k, fn in groups.items():
        group_data[k] = add_attr(fn(group_data), "source", k)
    
    ds = xr.merge(list(group_data.values()), join="inner")
    ds = add_meta_data(ds,meta_data)
    ds = add_uniq_id(ds, path)
    return ds


def fix_filter_variables(ds_dict):
    ds = ds_dict["FilterVariables"]
    retr_header_ds = ds_dict["RetrievalHeader"]
    ds = ds.rename(
        {"phony_dim_0":"sounding"}
    )
    ds = ds.assign_coords({"sounding":retr_header_ds["sounding_id"].values})
    ds = ds.rename({var: f"filter_{var}" for var in ds.data_vars})
    return ds

def fix_retrieval_header(ds_dict):
    ds = ds_dict["RetrievalHeader"]
    ds = ds.rename(
        {"phony_dim_0":"sounding"}
    )
    ds = ds.assign_coords({"sounding":ds["sounding_id"].values})
    return ds

def fix_retrieval_results(ds_dict):
    ds = ds_dict["RetrievalResults"]
    retr_header_ds = ds_dict["RetrievalHeader"]
    ds = ds.rename(
        {
            "phony_dim_0":"sounding", 
            "phony_dim_1":"state_vector_names",
            "phony_dim_2":"state_vector_names_",
            "phony_dim_3":"pressure_level"
        }
    )
    ds = ds.assign_coords({
        "sounding":retr_header_ds["sounding_id"].values,
        "state_vector_names":ds["state_vector_names"].values,
        "state_vector_names_":ds["state_vector_names"].values,
        "pressure_level":np.array([f"{i}" for i in range(1, 21)])
    })
    if "retrieved_state_vector" in ds.data_vars:
        for i in range(1,21):
            var_name = f"retrieved_state_vector_{i:02d}"
            ds[var_name] = (("sounding",), ds["retrieved_state_vector"][:, i-1].values)

    # Extract CO2 profile covariance (20×20 from top-left of 49×49)
    if "retrieved_state_vector_covariance" in ds.data_vars:
        full_cov = ds["retrieved_state_vector_covariance"].values
        co2_cov = full_cov[:, :20, :20]
        ds["retrieved_co2_profile_covariance"] = (
            ("sounding", "profile_level_i", "profile_level_j"), 
            co2_cov
        )
    return ds

def fix_spectral_parameters(ds_dict):
    ds = ds_dict["SpectralParameters"]
    retr_header_ds = ds_dict["RetrievalHeader"]
    
    ds = ds.rename(
        {"phony_dim_0":"sounding", "phony_dim_1":"wavelength_index"}
    )

    ds = ds.assign_coords(
        {
            "sounding":retr_header_ds['sounding_id'].values, 
            "wavelength_index":np.arange(ds['wavelength'].values.shape[-1])
        }
    )
    return ds

def fix_state_vector(ds_dict):
    ds = ds_dict["StateVector"]
    retr_res_ds = ds_dict["RetrievalResults"]
    retr_header_ds = ds_dict["RetrievalHeader"]
    
    ds = ds.rename(
        {"phony_dim_0":"sounding", "phony_dim_1":"state_vector_names"}
    )

    ds = ds.assign_coords(
        {
            "sounding":retr_header_ds['sounding_id'].values, 
            "state_vector_names":retr_res_ds["state_vector_names"].values
        }
    )
    if "true_state_vector" in ds.data_vars:
        for i in range(1,21):
            var_name = f"true_state_vector_{i:02d}"
            ds[var_name] = (("sounding",), ds["true_state_vector"][:, i-1].values)
    return ds

def get_meta_data(path:Path)->pd.DataFrame:
    """Get the meta data for a cluster."""
    df = None
    for repeat in path.iterdir():
        if repeat.suffix=='.csv':
            df = pd.read_csv(str(repeat))
    return df

def get_property(path:Path, prop:str="ref"):
    """Get a property using the path to the data (e.g. clst, ref)."""
    pattern = f"{prop}([0-9]+)"
    return int(re.search(pattern, path.stem).group(1))

def add_var(ds, key, value, coordinate="sounding"):
    """Add a single variable to the dataset."""
    num_sounding = ds.sizes[coordinate]
    if not isinstance(value, list):

        value_ = [value for j in range(num_sounding)]
        value = value_
    ds[key] = ((coordinate,), value)
    return ds

def add_uniq_id(ds, path):
    clst_id = get_property(path, "clst")
    ref_id = get_property(path, "ref")
    uniq_id = [f"clst{clst_id:04d}_ref{ref_id:04d}_rep{i:04d}" for i in range(1,ds.sizes['sounding']+1)]

    return add_var(ds, "ID", uniq_id, coordinate="sounding")

def add_meta_data(ds, meta_data, meta_columns = META_COLUMNS):
    """Add META_COLUMNS from metadata to dataset."""
    for column in meta_columns:
        ds = add_var(ds, column, meta_data[column], coordinate="sounding")
    return ds

def add_attr(ds, attr_name, attr_value):
    """Add an attribute to all the variables in a dataset."""
    for var_name, dataarray in ds.data_vars.items():
        ds[var_name].attrs[attr_name] = attr_value  
    return ds

