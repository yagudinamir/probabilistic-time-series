import torch
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import sklearn.datasets
import math
from scipy.stats import levy_stable


def get_bimodal_dataset(n_datapoints, seed):
    rng = np.random.RandomState(seed)
    loc_a, loc_b = 0, 0
    X = np.linspace(-6, 6, n_datapoints)
    y = np.linspace(-6, 6, n_datapoints) ** 2

    perm = rng.permutation(X.shape[0])
    noise = np.concatenate(
        [
            rng.normal(0, 20, size=int(n_datapoints / 2)),
            rng.normal(0, 1, size=int(n_datapoints / 2)),
        ]
    )
    noise = noise[perm]
    y_noisy = y + noise
    X = X.reshape(-1, 1)

    return X, y_noisy


def get_cubic_dataset(n_datapoints, seed):
    rng = np.random.RandomState(seed)
    X = np.linspace(-4, 4, n_datapoints)
    y = np.linspace(-4, 4, n_datapoints) ** 3
    noise = rng.normal(0, 3, size=n_datapoints)
    y_noisy = y + noise
    X = X.reshape(-1, 1)

    return X, y_noisy


def get_linear_dataset(n_datapoints, seed):
    rng = np.random.RandomState(seed)
    X = np.linspace(-4, 4, n_datapoints)
    y = np.linspace(-4, 4, n_datapoints) * 0.5
    noise = rng.normal(0, math.sqrt(2), size=n_datapoints)
    y_noisy = y + noise
    X = X.reshape(-1, 1)

    return X, y_noisy


def get_sklearn_synthetic(n_datapoints):
    return sklearn.datasets.make_regression(
        n_samples=n_datapoints, n_features=15, noise=3.0, random_state=0
    )


def get_simulated_dataloaders(
    dataset_type="cubic",
    split_seed=0,
    test_fraction=0.3,
    train_frac=1.0,
    batch_size=128,
):
    r"""
    Returns Pytorch dataloaders for a simulated regression problem:
       y = X + eps where eps is sampled from a noise distribution.
    Arguments:
        dataset_config (dictionary): specifies the noise distribution

    Returns:
        train (Pytorch dataloader): training data
        val (Pytorch dataloader): validation data
        test (Pytorch dataloader): test data
    """
    n_datapoints = 5000
    restrict_train_frac = train_frac
    seed = split_seed
    if dataset_type == "cubic":
        X, y_noisy = get_cubic_dataset(n_datapoints, seed)
    elif dataset_type == "linear":
        X, y_noisy = get_linear_dataset(n_datapoints, seed)
    elif dataset_type == "bimodal":
        X, y_noisy = get_bimodal_dataset(n_datapoints, seed)
    elif dataset_type == "sklearn":
        X, y_noisy = get_sklearn_synthetic(n_datapoints, seed)

    rs = np.random.RandomState(seed)
    permutation = rs.permutation(X.shape[0])

    size_train = int(np.round(X.shape[0] * (1 - test_fraction)))
    index_train = permutation[0:size_train]
    index_test = permutation[size_train:]

    X_train = X[index_train, :]
    X_test = X[index_test, :]

    y_train = y_noisy[index_train, None]
    y_test = y_noisy[index_test, None]

    val_fraction = 0.3
    size_train = int(np.round(X_train.shape[0] * (1 - val_fraction)))
    permutation = rs.permutation(X_train.shape[0])
    index_train = permutation[0:size_train]
    index_val = permutation[size_train:]

    X_new_train = X_train[index_train, :]
    X_val = X_train[index_val, :]
    y_new_train = y_train[index_train]
    y_val = y_train[index_val]

    print("Done loading sim dataset")

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

    n_train = int(restrict_train_frac * len(X_new_train))
    train = TensorDataset(
        torch.Tensor(X_new_train[:n_train]).type(torch.float64),
        torch.Tensor(y_new_train[:n_train]).type(torch.float64),
    )

    val = TensorDataset(
        torch.Tensor(X_val).type(torch.float64),
        torch.Tensor(y_val).type(torch.float64),
    )

    test = TensorDataset(
        torch.Tensor(X_test).type(torch.float64),
        torch.Tensor(y_test).type(torch.float64),
    )

    train_loader = DataLoader(train, batch_size=len(train), shuffle=True)
    val_loader = DataLoader(val, batch_size=len(val), shuffle=False)
    test_loader = DataLoader(test, batch_size=len(test), shuffle=False)
    return (
        train_loader,
        val_loader,
        test_loader,
        torch.tensor(y_train_scale),
        X[0].shape,
    )
