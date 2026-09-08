"""processing.py
Functions for processing data.
"""
import logging
logger = logging.getLogger(__name__)

import xarray as xr
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from pathlib import Path

def skip_empty_ds(func):
    """Decorator that skips function execution if the input dataset is empty (has no dimensions)."""
    def wrapped(ds, *args, **kwargs):
        if ds.dims:
            return func(ds, *args, **kwargs)
        else:
            return ds
    return wrapped 

@skip_empty_ds
def embed_variable(ds:xr.Dataset, variable:str, net:callable)-> xr.Dataset|None:
    """Get embeddings for variable using neural net.
    """
    pass

@skip_empty_ds
def filter_variable(ds:xr.Dataset, variable:str, condition:callable)-> xr.Dataset | None:
    """Filters a dataset by applying a condition to a specified variable.

    This function is designed for use with xarray-datatree's `map_over_datasets`, enabling 
    flexible filtering of datasets within a DataTree structure.

    :param ds: The input dataset to be filtered.
    :type ds: xr.Dataset
    :param variable: The name of the variable in the dataset to apply the condition on.
    :type variable: str
    :param condition: A function that takes the variable's values and returns a boolean array 
        indicating where the condition is met.
    :type condition: callable

    :returns: The filtered dataset if the variable has non-zero size; otherwise, None.
    :rtype: xr.Dataset or None

    :example:
        dt.map_over_datasets(filter_variable, "Region", lambda v: np.isclose(v, 6, atol=1e-3))
    """


    selected = condition(ds[variable])
    new_ds = ds.where(selected, drop = True)
    
    if new_ds[variable].size == 0:
        return new_ds
    else:
        return new_ds

@skip_empty_ds
def add_variable(ds:xr.Dataset,variable:str, coordinates:tuple, op:callable)-> xr.Dataset | None:
    """Add a variable by calling an operation on the original dataset.

    This function is designed for use with xarray-datatree's `map_over_datasets`, enabling 
    flexible addition of new variables to the dataset.

    :param ds: The input dataset to be filtered.
    :type ds: xr.Dataset
    :param coordinates: The coordinates of the variable being added.
    :type coordinates: tuple
    :param op: A function that computes the variable.
    :type op: callable

    :returns: The filtered dataset if the variable has non-zero size; otherwise, None.
    :rtype: xr.Dataset or None

    :example:
        dt.map_over_datasets(add_variable, "split", lamba: ['Train' for i in range(ds.sizes[0])])
    """

    new_ds = deepcopy(ds)
    new_ds[variable] = xr.DataArray(
        data = op(new_ds),
        coords = {coord:new_ds[coord] for coord in coordinates}
    )
    return new_ds

@skip_empty_ds
def extract_variables(ds:xr.Dataset, variables:list):
    return ds[variables]



def flatten_tree(dt:xr.DataTree):
    """Flattens an xarray DataTree into a dictionary of NumPy arrays.

    This function walks through all leaf nodes in the DataTree, collects
    data from each dataset, and concatenates data arrays along the first axis.

    :param dt: The input DataTree to be flattened.
    :type dt: xr.DataTree

    :return: A dictionary where keys are variable names and values are concatenated 
             NumPy arrays from all datasets in the tree.
    :rtype: dict[str, np.ndarray]
    """
    res_dict = defaultdict(list)

    for tree in dt.subtree:
        if tree.is_leaf:
            if tree.has_data:
                ds = tree.dataset
                for variable in ds.data_vars:
                    res_dict[variable].append(ds[variable].data)
    for variable in res_dict.keys():
        res_dict[variable] = np.concatenate(res_dict[variable], axis = 0)

    return res_dict



def full_pipe(ds, variables, target, coord, mask):
    return ds.pipe(
        extract_variables, variables = variables
    ).pipe(
        filter_variable, variable = target, condition = lambda z : ~np.isnan(z)
    ).pipe(
        drop_along_coord, coord=coord, mask=mask
    ).pipe(
        fill
    )

@skip_empty_ds
def fill(ds):
    return ds.fillna(1.0)

@skip_empty_ds
def get_missing(ds, variable):
    return ds[variable].isnull().to_dataset(name = f"no_{variable}")

@skip_empty_ds
def drop_along_coord(ds,coord, mask):
    labels = {coord:ds[coord].values[mask]}
    return ds.drop_sel(labels)

def save_data(dt, path, aux_vars,  data_vars):
    path.mkdir(parents=True, exist_ok=True)
    flat_dt = flatten_tree(dt)
    aux_data = {}
    for variable in aux_vars:
        aux_data[variable] = flat_dt[variable]
    aux_data = pd.DataFrame(aux_data)

    spectra_data = []
    for variable in data_vars:
        spectra_data.append(flat_dt[variable])
    spectra_data = np.stack(spectra_data, axis=2)

    for data, uniq_id in zip(spectra_data, aux_data["ID"]):
        np.save(
            (path / uniq_id).with_suffix(".npy"), 
            data
        )
    aux_data.to_csv(
        path / 'aux_data.csv', 
        index = False
    )

def save_data_with_covariance(dt, path, aux_vars, data_vars, covariance_var="retrieved_co2_profile_covariance"):
    """Save spectral data, auxiliary CSV, and covariance matrices"""
    path.mkdir(parents=True, exist_ok=True)
    flat_dt = flatten_tree(dt)
    
    # Extract auxiliary data
    aux_data = {}
    for variable in aux_vars:
        aux_data[variable] = flat_dt[variable]
    aux_data = pd.DataFrame(aux_data)
    
    # Extract spectra
    spectra_data = []
    for variable in data_vars:
        spectra_data.append(flat_dt[variable])
    spectra_data = np.stack(spectra_data, axis=2)
    
    # Extract covariance if available
    has_covariance = covariance_var in flat_dt
    if has_covariance:
        covariance_data = flat_dt[covariance_var]  # (N, 20, 20)
        logger.info(f"Saving covariance matrices with shape: {covariance_data.shape}")
        
        for data, cov, uniq_id in zip(spectra_data, covariance_data, aux_data["ID"]):
            np.save((path / uniq_id).with_suffix(".npy"), data)
            np.save((path / f"{uniq_id}_cov").with_suffix(".npy"), cov)
    else:
        logger.warning(f"Covariance variable '{covariance_var}' not found in data")
        for data, uniq_id in zip(spectra_data, aux_data["ID"]):
            np.save((path / uniq_id).with_suffix(".npy"), data)
    
    aux_data.to_csv(path / 'aux_data.csv', index=False)


