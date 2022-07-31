from pytorch_lightning import Trainer, callbacks
import torch
from modules import (
    PointRecalibrationModel,
    AverageRecalibrationModel,
    DistributionRecalibrationModel,
    GaussianNLLModel,
    GaussianLaplaceMixtureNLLModel,
    NoRecalibrationModel,
    IterativePointRecalibrationModel,
    IterativeAlphaPointRecalibrationModel,
)
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from data_loaders import (
    get_credit_regression_dataloader,
    get_uci_dataloaders,
    get_satellite_dataloaders,
    get_recalibration_dataloaders,
    get_mimic_dataloaders,
)
from torch.utils.data import TensorDataset
from distributions import GaussianDistribution, GaussianLaplaceMixtureDistribution
import argh
from reporting import report_recalibration_results
import math

from custom_probabilistic_time_series import get_nasdaq_dataloaders, get_nasdaq_model_predictions


def get_dataset(dataset, seed, train_frac, combine_val_train, resnet=False):
    batch_size = None
    if dataset in ["nasdaq"]:
        train, val, test, in_size, target_size, y_scale = get_nasdaq_dataloaders(
            dataset,
            split_seed=seed,
            test_fraction=0.3,
            batch_size=batch_size,
            train_frac=train_frac,
            combine_val_train=combine_val_train,
        )
    elif dataset in [
        "satellite",
        "combined_satellite",
        "uganda",
        "tanzania",
        "rwanda",
        "malawi",
        "mozambique",
        "zimbabwe",
    ]:
        train, val, test, in_size, output_size, y_scale = get_satellite_dataloaders(
            name=dataset,
            split_seed=seed,
            batch_size=32,
            combine_val_train=combine_val_train,
            resnet=resnet,
        )
    elif dataset in ["cubic"]:
        train, val, test, y_scale, in_size = get_simulated_dataloaders(
            dataset,
            split_seed=seed,
            test_fraction=0.3,
            batch_size=batch_size,
            train_frac=train_frac,
        )
    elif dataset in ["mimic_los"]:
        train, val, test, in_size, output_size, y_scale = get_mimic_dataloaders(
            dataset,
            split_seed=seed,
            test_fraction=0.3,
            batch_size=None,
            train_frac=train_frac,
        )

    elif dataset in ["credit"]:
        (train, val, test, in_size, output_size, y_scale,) = get_credit_regression_dataloader(
            split_seed=seed,
            batch_size=batch_size,
        )

    else:
        train, val, test, in_size, output_size, y_scale = get_uci_dataloaders(
            dataset,
            split_seed=seed,
            test_fraction=0.3,
            batch_size=batch_size,
            train_frac=train_frac,
            combine_val_train=combine_val_train,
        )

    return train, val, test, in_size, y_scale


def get_baseline_model_predictions(model, dist_class, train, val, test, cuda=False):
    print("getting baseline model preds")
    if cuda:
        device = torch.device("cuda")
        model.to(device)
        print("cuda")
    model.eval()

    def dataset_dist(data):
        ys = []
        all_params = []
        assert len(data) == 1  # somehow length of dataset is 1
        for batch in data:
            x, y = batch
            params = model(x)
            print(x.shape, y.shape, len(params), params[0].shape, params[1].shape)
            if cuda:
                x = x.to(device)
                y = y.to(device)
                params = [param.flatten().detach() for param in params]
                y = y.flatten()
            else:
                params = [param.detach().cpu().flatten() for param in params]
                y = y.flatten().detach().cpu()

            dist = dist_class(tuple(params))
            return dist, y

    train_dist, y_train = dataset_dist(train)
    print("got train dist")
    if val:
        val_dist, y_val = dataset_dist(val)
    else:
        val_dist = None
        y_val = None
    test_dist, y_test = dataset_dist(test)

    return train_dist, y_train, val_dist, y_val, test_dist, y_test


def get_baseline_model_predictions_resnet(model, dist_class, train, val, test, cuda=False):
    print("getting baseline model preds resnet")
    if cuda:
        device = torch.device("cuda")
        model.to(device)
        print("cuda")
    model.eval()

    def dataset_dist(data):
        ys = []
        all_params = []
        counter = 0
        for batch in data:
            x, y = batch
            params = model(x)
            if cuda:
                x = x.to(device)
                y = y.to(device)
                params = [param.flatten().detach() for param in params]
                y = y.flatten()
            else:
                params = [param.detach().cpu().flatten() for param in params]
                y = y.flatten().detach().cpu()
            all_params.append(params)
            ys.append(y)
            counter += 1
        print("HELLO")
        print("combining")
        params = []
        for i in range(len(all_params[0])):
            params.append(torch.cat([all_params[j][i] for j in range(counter)]))
        y = torch.cat([ys[j] for j in range(counter)])
        dist = dist_class(tuple(params))
        return dist, y

    train_dist, y_train = dataset_dist(train)
    print("got train dist")
    if val:
        val_dist, y_val = dataset_dist(val)
    else:
        val_dist = None
        y_val = None
    test_dist, y_test = dataset_dist(test)
    return train_dist, y_train, val_dist, y_val, test_dist, y_test


def train_recalibration_model(model, epochs, logname=None, actual_datasets=None):
    print(model.__class__.__name__)
    train, val, test = actual_datasets  # hack so can use pytorch lightning for training

    if "PointRecalibrationModel" == model.__class__.__name__:
        assert False  # temporary solution for nasdaq dataset
        logger = TensorBoardLogger(save_dir="runs", name="logs/{}".format(logname))
        #        early_stop_callback = callbacks.early_stopping.EarlyStopping(monitor='point_calibration_error_uniform_mass', min_delta=0.00, patience=50, verbose=False, mode='min')

        if val:
            checkpoint_callback = callbacks.model_checkpoint.ModelCheckpoint(
                "recalibration_models/{}/".format(logname),
                monitor="val_loss",
                save_top_k=1,
                mode="min",
            )

            trainer = Trainer(
                gpus=1,
                checkpoint_callback=checkpoint_callback,
                #            callbacks = [early_stop_callback],
                max_epochs=epochs,
                logger=logger,
                check_val_every_n_epoch=1,
                log_every_n_steps=1,
            )

            trainer.fit(model, train_dataloader=train, val_dataloaders=val)

        else:
            trainer = Trainer(
                gpus=1,
                #            checkpoint_callback=checkpoint_callback,
                #            callbacks = [early_stop_callback],
                max_epochs=epochs,
                logger=logger,
                check_val_every_n_epoch=1,
                log_every_n_steps=1,
            )

            trainer.fit(model, train_dataloader=train)
        trainer.test(test_dataloaders=test)
    elif (
        "Distribution" in model.__class__.__name__
        or "Average" in model.__class__.__name__
        or "IterativePoint" in model.__class__.__name__
        or "IterativeAlphaPoint" in model.__class__.__name__
    ):
        model.training_step()
        if val:
            val_outputs = model.validation_step()
        test_outputs = model.testing_step()
        model.test_epoch_end([test_outputs])
    else:
        if val:
            val_outputs = model.validation_step()
        test_outputs = model.testing_step()
        model.test_epoch_end([test_outputs])
    return model


def get_base_preds(model, dist_class, train, val, test, cuda, resnet):
    if model == "nasdaq":
        return get_nasdaq_model_predictions(model, dist_class, train, val, test, cuda=cuda)
    if resnet == False:
        dist_datasets = get_baseline_model_predictions(model, dist_class, train, val, test, cuda=cuda)
    else:
        dist_datasets = get_baseline_model_predictions_resnet(model, dist_class, train, val, test, cuda=cuda)
    return dist_datasets


@argh.arg("--resnet", default=False)
@argh.arg("--seed", default=0)
@argh.arg("--dataset", default="protein")
@argh.arg("--loss", default="gaussian_nll")
@argh.arg("--save", default="protein_evaluation")
@argh.arg("--posthoc_recalibration", default=None)
@argh.arg("--train_frac", default=1.0)
@argh.arg("--combine_val_train", default=False)
@argh.arg("--val_only", default=False)
@argh.arg("--cuda", default=False)
@argh.arg("--save_dir", default="results")
## Recalibration parameters
@argh.arg("--num_layers", default=0)
@argh.arg("--n_dim", default=100)
@argh.arg("--epochs", default=500)
@argh.arg("--n_bins", default=None)
@argh.arg("--flow_type", default=None)
@argh.arg("--learning_rate", default=1e-3)
def main(
    dataset="protein",
    seed=0,
    save="real",
    loss="point_calibration_loss",
    posthoc_recalibration=None,
    train_frac=1.0,
    num_layers=2,
    n_dim=100,
    epochs=500,
    n_bins=None,
    flow_type=None,
    learning_rate=1e-3,
    combine_val_train=False,
    val_only=False,
    save_dir="results",
    cuda=False,
    resnet=False,
):

    train, val, test, in_size, y_scale = get_dataset(dataset, seed, train_frac, combine_val_train, resnet)

    if val_only:
        train = val
    if n_bins != None:
        n_bins = int(n_bins)

    if loss == "gaussian_nll":
        model_class = GaussianNLLModel
        dist_class = GaussianDistribution
        n_in = 3
    elif loss == "gaussian_laplace_mixture_nll":
        model_class = GaussianLaplaceMixtureNLLModel
        dist_class = GaussianLaplaceMixtureDistribution
        n_in = 6
    if resnet:
        model_path = "models/{}_{}_resnet_seed_{}.ckpt".format(dataset, loss, seed)
    else:
        model_path = "models/{}_{}_seed_{}.ckpt".format(dataset, loss, seed)

    if "point" == posthoc_recalibration:
        recalibration_parameters = {
            "num_layers": num_layers,
            "n_dim": n_dim,
            "epochs": epochs,
            "n_bins": n_bins,
            "flow_type": flow_type,
            "learning_rate": learning_rate,
        }
    elif "iterative_point" == posthoc_recalibration or "iterative_alpha_point":
        recalibration_parameters = {"n_bins": n_bins, "num_layers": num_layers}
    elif "distribution" == posthoc_recalibration:
        recalibration_parameters = {"n_bins": n_bins}
    else:
        recalibration_parameters = None

    if dataset != "nasdaq":
        model = model_class.load_from_checkpoint(model_path, input_size=in_size[0], y_scale=y_scale, resnet=resnet)
        if resnet:
            print(resnet)
        print("loaded model")
    else:
        print("using nasdaq dataset")
        model = "nasdaq"

    if posthoc_recalibration == "point":
        assert dataset != "nasdaq"
        if val_only:
            logname = "{}_val_{}_sigmoid_{}layers_{}_{}dim_{}bins_{}epochs_{}lr_{}".format(
                dataset,
                loss,
                num_layers,
                flow_type,
                n_dim,
                n_bins,
                epochs,
                learning_rate,
                seed,
            )
        else:
            logname = "{}_{}_sigmoid_{}layers_{}_{}dim_{}bins_{}epochs_{}lr_{}".format(
                dataset,
                loss,
                num_layers,
                flow_type,
                n_dim,
                n_bins,
                epochs,
                learning_rate,
                seed,
            )
        dist_datasets = get_base_preds(model, dist_class, train, val, test, cuda=False, resnet=resnet)

        print("got baseline preds")
        if n_bins == None:
            n_bins = int(math.sqrt(dist_datasets[1].shape[0]))
            recalibration_parameters["n_bins"] = n_bins

        dataloaders = get_recalibration_dataloaders(*dist_datasets)
        print("got dataloaders")
        del model
        recalibration_model = PointRecalibrationModel(
            dist_datasets,
            n_in=n_in,
            n_layers=num_layers,
            n_dim=n_dim,
            n_bins=n_bins,
            flow_type=flow_type,
            y_scale=y_scale,
        )
        recalibration_model = train_recalibration_model(
            recalibration_model, epochs, logname=logname, actual_datasets=dataloaders
        )
    elif posthoc_recalibration == "average":
        dist_datasets = get_base_preds(model, dist_class, train, val, test, cuda=False, resnet=resnet)
        del model
        recalibration_model = AverageRecalibrationModel(dist_datasets, y_scale=y_scale)
        recalibration_model = train_recalibration_model(
            recalibration_model, 1, logname=None, actual_datasets=(train, val, test)
        )
    elif posthoc_recalibration == "iterative_point":
        dist_datasets = get_base_preds(model, dist_class, train, val, test, cuda=False, resnet=resnet)
        del model
        if n_bins == None:
            n_bins = int(math.sqrt(dist_datasets[1].shape[0]))
            recalibration_parameters["n_bins"] = n_bins
        recalibration_model = IterativePointRecalibrationModel(
            dist_datasets, y_scale=y_scale, n_bins=n_bins, num_layers=num_layers
        )
        recalibration_model = train_recalibration_model(
            recalibration_model, 1, logname=None, actual_datasets=(train, val, test)
        )
    elif posthoc_recalibration == "iterative_alpha_point":
        dist_datasets = get_base_preds(model, dist_class, train, val, test, cuda=False, resnet=resnet)
        del model
        if n_bins == None:
            n_bins = int(math.sqrt(dist_datasets[1].shape[0]))
            recalibration_parameters["n_bins"] = n_bins
        recalibration_model = IterativeAlphaPointRecalibrationModel(
            dist_datasets, y_scale=y_scale, n_bins=n_bins, num_layers=num_layers
        )
        recalibration_model = train_recalibration_model(
            recalibration_model, 1, logname=None, actual_datasets=(train, val, test)
        )

    elif posthoc_recalibration == "distribution":
        dist_datasets = get_base_preds(model, dist_class, train, val, test, cuda=False, resnet=resnet)
        if n_bins == None and loss == "gaussian_nll":
            n_bins = int(math.sqrt(dist_datasets[1].shape[0]))
            recalibration_parameters["n_bins"] = n_bins
        elif n_bins == None:
            n_bins = min(int(math.sqrt(dist_datasets[1].shape[0])), 20)
            recalibration_parameters["n_bins"] = n_bins
        del model
        recalibration_model = DistributionRecalibrationModel(dist_datasets, y_scale=y_scale, n_bins=n_bins)
        recalibration_model = train_recalibration_model(
            recalibration_model, 1, logname=None, actual_datasets=(train, val, test)
        )
    elif posthoc_recalibration == None:
        dist_datasets = get_base_preds(model, dist_class, train, val, test, cuda=False, resnet=resnet)
        recalibration_model = NoRecalibrationModel(dist_datasets, y_scale=y_scale)
        recalibration_model = train_recalibration_model(
            recalibration_model, 1, logname=None, actual_datasets=(train, val, test)
        )

    report_recalibration_results(
        recalibration_model,
        dataset,
        train_frac,
        loss,
        seed,
        posthoc_recalibration,
        recalibration_parameters,
        save,
        save_dir,
    )
    return recalibration_model


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
