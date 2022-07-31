import torch
from sklearn.isotonic import IsotonicRegression
import numpy as np
import math
from metrics import Metrics
from distributions import FlexibleDistribution
from composition import AlphaRecalibrationLayer
from tqdm import tqdm

RANGE = [-10, 10]
RESOLUTION = 8000


class IterativeAlphaPointRecalibrationModel:
    def __init__(self, datasets, y_scale, n_bins, num_layers):
        (
            self.train_dist,
            self.y_train,
            self.val_dist,
            self.y_val,
            self.test_dist,
            self.y_test,
        ) = datasets
        self.y_scale = y_scale
        self.n_bins = n_bins
        self.num_layers = num_layers

    def training_step(self):
        bin_size = int(self.y_train.shape[0] / self.n_bins)
        current_dist = self.train_dist
        model = []
        indices = torch.ones((50, self.n_bins))
        for i in tqdm(range(self.num_layers)):
            metrics = Metrics(
                current_dist,
                self.y_train,
                self.y_scale,
                discretization=self.n_bins,
            )
            errs, thresholds, alphas = metrics.decision_unbiasedness()
            idx = torch.where(errs == torch.max(errs[indices.bool()]))
            choice = (torch.rand(1) * len(idx[0])).long()
            #            indices[idx[0][choice], idx[1][choice]] = 0.
            t_idx = idx[0][choice]
            threshold = torch.Tensor([thresholds[t_idx]])
            errs2, _, _ = metrics.point_calibration_error_uniform_mass_errs()
            print("pce", torch.max(errs2), torch.mean(errs2))
            print(threshold, idx)
            current_train_forecasts = current_dist.cdf(self.y_train.flatten())
            quantile_threshold = current_dist.cdf(threshold).flatten()
            alpha = alphas[idx][choice].item()
            all_subgroups = (
                torch.where(quantile_threshold <= alpha),
                torch.where(quantile_threshold > alpha),
            )
            iso_reg_models = []
            for j in range(len(all_subgroups)):
                true_vals = current_train_forecasts[all_subgroups[j]]
                sorted_vals = torch.sort(true_vals.flatten())[0].detach().cpu().numpy()
                Y = np.array(
                    [(k + 1) / (len(sorted_vals) + 2) for k in range(len(sorted_vals))]
                )
                Y = np.insert(np.insert(Y, 0, 0), len(Y) + 1, 1)
                sorted_forecasts = np.insert(
                    np.insert(sorted_vals, 0, 0), len(sorted_vals) + 1, 1
                )
                iso_reg = IsotonicRegression().fit(sorted_forecasts.flatten(), Y)
                iso_reg_models.append(iso_reg)
            r = AlphaRecalibrationLayer(iso_reg_models, alpha, threshold)
            model.append(r)
            current_dist = output_distribution_single_layer(current_dist, r)

        self.model = model
        cdfs = []
        y = torch.linspace(RANGE[0], RANGE[1], RESOLUTION)
        params = self.train_dist.params
        dist = output_distribution_all_layers(self.train_dist, self.model)
        metrics = Metrics(dist, self.y_train, self.y_scale)
        dic = metrics.get_metrics(decision_making=True)
        #        setattr(
        #            self,
        #            "train_point_calibration_error_uniform_mass",
        #            dic["point_calibration_error_uniform_mass"].item(),
        #        )
        #        setattr(
        #            self, "train_point_calibration_error", dic["point_calibration_error"].item()
        #        )
        #        setattr(self, "train_true_vs_pred_loss", dic["true_vs_pred_loss"].item())

        print("Done train")

    def testing_step(self):
        print("Test...")
        cdfs = []
        y = torch.linspace(RANGE[0], RANGE[1], RESOLUTION)
        params = self.test_dist.params
        dist = output_distribution_all_layers(self.test_dist, self.model)
        metrics = Metrics(dist, self.y_test, self.y_scale)
        dic = metrics.get_metrics(decision_making=True)
        return dic

    def test(self):
        print("Test...")
        cdfs = []
        y = torch.linspace(RANGE[0], RANGE[1], RESOLUTION)
        params = self.test_dist.params
        dist = output_distribution_all_layers(self.test_dist, self.model)
        return dist

    def validation_step(self):
        print("Validation...")

        cdfs = []
        y = torch.linspace(RANGE[0], RANGE[1], RESOLUTION)
        params = self.val_dist.params
        dist = output_distribution_all_layers(self.val_dist, self.model)
        metrics = Metrics(dist, self.y_val, self.y_scale)
        dic = metrics.get_metrics(decision_making=True)
        #        setattr(
        #            self,
        #            "val_point_calibration_error_uniform_mass",
        #            dic["point_calibration_error_uniform_mass"].item(),
        #        )
        #        setattr(
        #            self, "val_point_calibration_error", dic["point_calibration_error"].item()
        #        )
        #        setattr(self, "val_true_vs_pred_loss", dic["true_vs_pred_loss"].item())
        return dic

    def test_epoch_end(self, outputs):
        for key in outputs[0]:
            if key not in ["all_err", "all_loss", "all_y0", "all_c"]:
                cal = torch.stack([x[key] for x in outputs]).mean()
                setattr(self, key, float(cal))
            else:
                setattr(self, key, outputs[0][key])


def output_distribution_single_layer(dist, r):
    cdfs = []
    y = torch.linspace(RANGE[0], RANGE[1], RESOLUTION)
    if "Flexible" not in dist.__class__.__name__:
        n_dist = dist.params[0].shape[0]

        for i in range(n_dist):
            params = dist.params
            sub_params = tuple([params[j][[i]] for j in range(len(params))])
            small_test_dist = dist.__class__(sub_params)
            res = r.cdf(small_test_dist, y)
            cdfs.append(res.flatten())
        ranking = torch.tensor(cdfs)
    else:
        n_dist = dist.cdfs.shape[0]
        for i in range(n_dist):
            small_test_dist = dist.__class__((dist.xs, dist.cdfs[[i]]))
            res = r.cdf(small_test_dist, y)
            cdfs.append(res.flatten())
        ranking = torch.tensor(cdfs)
    new_dist = FlexibleDistribution((y, ranking))
    return new_dist


def output_distribution_all_layers(dist, model):
    current_dist = dist
    for layer in model:
        current_dist = output_distribution_single_layer(current_dist, layer)
    return current_dist
