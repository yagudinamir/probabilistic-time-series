"""
IO module for UCI datasets for regression

Source: https://github.com/aamini/evidential-regression/blob/c0823f18ff015f5eb46a23f0039f4d62b76bc8d1/data_loader.py
"""
import numpy as np
import pandas as pd
import os
import h5py
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from sklearn.datasets import load_boston
from data_loaders.utils import get_dataloaders


def get_uci_datasets(name, split_seed=0, test_fraction=0.10, train_frac=1.0, combine_val_train=False):
    r"""
    Returns a UCI regression dataset in the form of numpy arrays.

    Arguments:
        name (str): name of the dataset
        split_seed (int): seed used to generate train/test split
        test_fraction (float): fraction of the dataset used for the test set


    Returns:
        X_train (numpy.ndarray): training features
        y_train (numpy.ndaray): training label
        X_test (numpy.ndarray): test features
        y_test (numpy.ndarray): test labels
        y_train_scale (float): standard deviation of training labels
    """
    # load full dataset
    load_funs = {
        "naval": _load_naval,
        "protein": _load_protein,
        "crime": _load_crime,
        "energy": _load_app_energy,
    }
    print("Loading dataset {}....".format(name))
    if name == "depth":
        (X_train, y_train), (X_test, y_test) = load_funs[name]()
        y_scale = np.array([[1.0]])
        return (X_train, y_train), (X_test, y_test), y_scale

    X, y = load_funs[name]()
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # We create the train and test sets with 90% and 10% of the data

    if split_seed == -1:  # Do not shuffle!
        permutation = range(X.shape[0])
    else:
        rs = np.random.RandomState(split_seed)
        permutation = rs.permutation(X.shape[0])

    size_train = int(np.round(X.shape[0] * (1 - test_fraction)))
    index_train = permutation[0:size_train]
    index_test = permutation[size_train:]

    X_train = X[index_train, :]
    X_test = X[index_test, :]
    if name == "depth":
        y_train = y[index_train]
        y_test = y[index_test]
    else:
        y_train = y[index_train, None]
        y_test = y[index_test, None]

    if train_frac != 1.0:
        rs = np.random.RandomState(split_seed)
        permutation = rs.permutation(X_train.shape[0])
        n_train = int(train_frac * len(X_train))
        X_train = X_train[:n_train]
        y_train = y_train[:n_train]

    if split_seed == -1:  # Do not shuffle!
        permutation = range(X_train.shape[0])
    else:
        rs = np.random.RandomState(split_seed)
        permutation = rs.permutation(X_train.shape[0])

    if combine_val_train:
        val_fraction = 0.0
    else:
        val_fraction = 0.10
    size_train = int(np.round(X_train.shape[0] * (1 - val_fraction)))
    index_train = permutation[0:size_train]
    index_val = permutation[size_train:]

    X_new_train = X_train[index_train, :]
    X_val = X_train[index_val, :]

    y_new_train = y_train[index_train]
    y_val = y_train[index_val]

    print("Done loading dataset {}".format(name))

    def standardize(data):
        mu = data.mean(axis=0, keepdims=1)
        scale = data.std(axis=0, keepdims=1)
        scale[scale < 1e-10] = 1.0

        data = (data - mu) / scale
        return data, mu, scale

    # Standardize
    X_new_train, x_train_mu, x_train_scale = standardize(X_new_train)
    X_test = (X_test - x_train_mu) / x_train_scale
    y_new_train, y_train_mu, y_train_scale = standardize(y_new_train)
    y_test = (y_test - y_train_mu) / y_train_scale
    X_val = (X_val - x_train_mu) / x_train_scale
    y_val = (y_val - y_train_mu) / y_train_scale

    train = TensorDataset(
        torch.Tensor(X_new_train).type(torch.float64),
        torch.Tensor(y_new_train).type(torch.float64),
    )

    val = TensorDataset(
        torch.Tensor(X_val).type(torch.float64),
        torch.Tensor(y_val).type(torch.float64),
    )

    test = TensorDataset(
        torch.Tensor(X_test).type(torch.float64),
        torch.Tensor(y_test).type(torch.float64),
    )
    in_size = X_train[0].shape
    target_size = y_train[0].shape

    return train, val, test, in_size, target_size, y_train_scale


def get_uci_dataloaders(
    name,
    split_seed=0,
    test_fraction=0.1,
    batch_size=None,
    train_frac=1.0,
    combine_val_train=False,
):
    r"""
    Returns a UCI regression dataset in the form of Pytorch dataloaders
    for train, validation, and test. Also returns the sizes of features and label
    and the standard deviation of the training labels.

    Arguments:
        name (str): name of the dataset
        split_seed (int): seed used to generate train/test split
        test_fraction (float): fraction of the dataset used for the test set

    Returns:
        train_loader (Pytorch dataloader): training data.
        val_loader (Pytorch dataloader): validation data.
        test_loader (Pytorch dataloader): test data.
        in_size (tuple): feature shape
        target_size (tuple): target shape
        y_train_scale (float): standard deviation of training labels.

    """
    train, val, test, in_size, target_size, y_train_scale = get_uci_datasets(
        name,
        split_seed=split_seed,
        test_fraction=test_fraction,
        train_frac=train_frac,
        combine_val_train=combine_val_train,
    )
    train_loader, val_loader, test_loader = get_dataloaders(train, val, test, batch_size)
    return train_loader, val_loader, test_loader, in_size, target_size, y_train_scale


#####################################
# individual data files             #
#####################################
vb_dir = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "data/uci")


def _load_naval():
    """
    http://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants
    Input variables:
    1 - Lever position(lp)[]
    2 - Ship speed(v)[knots]
    3 - Gas Turbine shaft torque(GTT)[kNm]
    4 - Gas Turbine rate of revolutions(GTn)[rpm]
    5 - Gas Generator rate of revolutions(GGn)[rpm]
    6 - Starboard Propeller Torque(Ts)[kN]
    7 - Port Propeller Torque(Tp)[kN]
    8 - HP Turbine exit temperature(T48)[C]
    9 - GT Compressor inlet air temperature(T1)[C]
    10 - GT Compressor outlet air temperature(T2)[C]
    11 - HP Turbine exit pressure(P48)[bar]
    12 - GT Compressor inlet air pressure(P1)[bar]
    13 - GT Compressor outlet air pressure(P2)[bar]
    14 - Gas Turbine exhaust gas pressure(Pexh)[bar]
    15 - Turbine Injecton Control(TIC)[ %]
    16 - Fuel flow(mf)[kg / s]
    Output variables:
    17 - GT Compressor decay state coefficient.
    18 - GT Turbine decay state coefficient.
    """
    data = np.loadtxt(os.path.join(data_dir, "naval/UCI CBM Dataset/data.txt"))
    X = data[:, :-2]
    y_compressor = data[:, -2]
    y_turbine = data[:, -1]
    return X, y_turbine


def _load_protein():
    """
    Physicochemical Properties of Protein Tertiary Structure Data Set
    Abstract: This is a data set of Physicochemical Properties of Protein Tertiary Structure.
    The data set is taken from CASP 5-9. There are 45730 decoys and size varying from 0 to 21 armstrong.
    TODO: Check that the output is correct
    Input variables:
        RMSD-Size of the residue.
        F1 - Total surface area.
        F2 - Non polar exposed area.
        F3 - Fractional area of exposed non polar residue.
        F4 - Fractional area of exposed non polar part of residue.
        F5 - Molecular mass weighted exposed area.
        F6 - Average deviation from standard exposed area of residue.
        F7 - Euclidian distance.
        F8 - Secondary structure penalty.
    Output variable:
        F9 - Spacial Distribution constraints (N,K Value).
    """
    data_file = os.path.join(data_dir, "protein/CASP.csv")
    data = pd.read_csv(data_file, sep=",")
    X = data.values[:, 1:]
    y = data.values[:, 0]
    return X, y


def _load_app_energy():
    def filter_nan(raw):
        idx = np.sum(np.isnan(raw), axis=1) == 0
        raw = raw[idx, :]
        return raw

    raw = np.genfromtxt(
        os.path.join(data_dir, "energy/energydata_complete.csv"),
        delimiter=",",
        skip_header=True,
    )

    raw = filter_nan(raw)
    X = raw[:, 1:]
    y = raw[:, 0]
    return X, y


def _load_crime():
    reader = open(os.path.join(data_dir, "crime/communities.data"))

    attributes = []
    while True:
        line = reader.readline().split(",")
        if len(line) < 128:
            break
        line = ["-1" if val == "?" else val for val in line]
        line = np.array(line[5:], dtype=np.float)
        attributes.append(line)
    reader.close()

    attributes = np.stack(attributes, axis=0)

    reader = open(os.path.join(data_dir, "crime/names"))
    names = []
    for i in range(128):
        line = reader.readline().split()[1]
        if i >= 5:
            names.append(line)
    names = np.array(names)

    y = attributes[:, -1:]
    attributes = attributes[:, :-1]
    selected = np.argwhere(np.array([np.min(attributes[:, i]) for i in range(attributes.shape[1])]) >= 0).flatten()
    X = attributes[:, selected]
    return X, y.flatten()


if __name__ == "__main__":
    load_funs = {
        "energy": _load_app_energy,
        "naval": _load_naval,
        "protein": _load_protein,
        "crime": _load_crime,
    }
    for k in load_funs:
        X, y = load_funs[k]()
        print(k, X.shape)
