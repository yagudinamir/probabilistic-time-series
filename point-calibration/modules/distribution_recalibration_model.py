import torch
import itertools
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import numpy as np
import math
from metrics import Metrics
from distributions import FlexibleDistribution

RANGE = [-10, 10]
RESOLUTION = 8000


class DistributionRecalibrationModel:
    def __init__(self, datasets, y_scale, n_bins):
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

    def training_step(self):
        train_forecasts = self.train_dist.cdf(self.y_train.flatten())

        def set_by_tuple(l, tupl, val):
            location = l
            for i, idx in enumerate(tupl):
                if i != len(tupl) - 1:
                    location = location[idx]
                else:
                    location[idx] = val

        #        n_params = len(self.train_dist.params)
        #        n = max(1, math.ceil(math.log(self.n_bins)/math.log(n_params)))
        n = self.n_bins
        parameter_bins = []
        grid_shape = []
        for i in range(len(self.train_dist.params)):
            param_bins = torch.linspace(
                torch.min(self.train_dist.params[i]),
                torch.max(self.train_dist.params[i]),
                n + 1,
            )
            parameter_bins.append(param_bins)
            grid_shape.append(n)

        self.parameter_bins = parameter_bins
        iso_reg_models = np.zeros(shape=grid_shape)
        iso_reg_models[iso_reg_models == 0] = None
        iso_reg_models = iso_reg_models.tolist()
        print("Distribution calibration...")
        cartesian_product = itertools.product(
            range(n), repeat=len(self.train_dist.params)
        )
        for tup in tqdm(cartesian_product):
            condition = torch.ones(self.train_dist.params[i].shape).bool()
            for i in range(len(self.train_dist.params)):
                condition = torch.logical_and(
                    condition,
                    (self.train_dist.params[i] < parameter_bins[i][tup[i] + 1])
                    & (self.train_dist.params[i] > parameter_bins[i][tup[i]]),
                )
            indices = torch.where(condition)
            true_vals = train_forecasts[indices[0]]
            sorted_forecasts = torch.sort(true_vals.flatten())[0].detach().cpu().numpy()
            Y = np.array(
                [
                    (k + 1) / (len(sorted_forecasts) + 2)
                    for k in range(len(sorted_forecasts))
                ]
            )
            Y = np.insert(np.insert(Y, 0, 0), len(Y) + 1, 1)
            sorted_forecasts = np.insert(
                np.insert(sorted_forecasts, 0, 0), len(sorted_forecasts) + 1, 1
            )

            iso_reg = IsotonicRegression().fit(sorted_forecasts.flatten(), Y)
            set_by_tuple(iso_reg_models, tup, iso_reg)
        self.iso_reg_models = iso_reg_models

    def testing_step(self):
        cdfs = []
        y = torch.linspace(RANGE[0], RANGE[1], RESOLUTION)

        def lookup_by_tuple(l, tupl):
            answer = l
            for i in tupl:
                answer = answer[i]
            return answer

        params = self.test_dist.params
        uncalibrated_cdf_count = 0
        ex_to_bin = {}
        print("Inference on test examples...")
        for i in range(self.test_dist.params[0].shape[0]):
            sub_params = tuple([params[j][[i]] for j in range(len(params))])
            small_test_dist = self.test_dist.__class__(sub_params)
            test_forecast = small_test_dist.cdf(y)
            idx = []
            for k in range(len(params)):
                idx.append(torch.where(self.parameter_bins[k] > sub_params[k]))
            updated = False
            can_recalibrate = True
            for sub_list in idx:
                if len(sub_list[0]) == 0 or sub_list[0][0] == 0:
                    can_recalibrate = False
                    break

            if can_recalibrate:
                indices = [idx[k][0][0].item() - 1 for k in range(len(idx))]
                iso_reg = lookup_by_tuple(self.iso_reg_models, tuple(indices))
                ex_to_bin[i] = tuple(indices)
                res = iso_reg.predict(test_forecast)
                cdfs.append(res.flatten())
                updated = True
            if not updated:
                cdfs.append(test_forecast.flatten().numpy())
                uncalibrated_cdf_count += 1
                print(uncalibrated_cdf_count)

        ranking = torch.tensor(cdfs)
        dist = FlexibleDistribution((y, ranking))
        metrics = Metrics(dist, self.y_test, self.y_scale)

        dic = metrics.get_metrics(decision_making=True)
        return dic

    def test(self):
        cdfs = []
        y = torch.linspace(RANGE[0], RANGE[1], RESOLUTION)

        def lookup_by_tuple(l, tupl):
            answer = l
            for i in tupl:
                answer = answer[i]
            return answer

        params = self.test_dist.params
        uncalibrated_cdf_count = 0
        print("Inference on test examples...")
        for i in range(self.test_dist.params[0].shape[0]):
            sub_params = tuple([params[j][[i]] for j in range(len(params))])
            small_test_dist = self.test_dist.__class__(sub_params)
            test_forecast = small_test_dist.cdf(y)
            idx = []
            for k in range(len(params)):
                idx.append(torch.where(self.parameter_bins[k] > sub_params[k]))
            updated = False
            can_recalibrate = True
            for sub_list in idx:
                if len(sub_list[0]) == 0 or sub_list[0][0] == 0:
                    can_recalibrate = False
                    break

            if can_recalibrate:
                indices = [idx[k][0][0].item() - 1 for k in range(len(idx))]
                iso_reg = lookup_by_tuple(self.iso_reg_models, tuple(indices))
                res = iso_reg.predict(test_forecast)
                cdfs.append(res.flatten())
                updated = True
            if not updated:
                cdfs.append(test_forecast.flatten().numpy())
                uncalibrated_cdf_count += 1
                print(uncalibrated_cdf_count)

        ranking = torch.tensor(cdfs)
        dist = FlexibleDistribution((y, ranking))
        return dist

    def validation_step(self):
        cdfs = []
        y = torch.linspace(RANGE[0], RANGE[1], RESOLUTION)

        def lookup_by_tuple(l, tupl):
            answer = l
            for i in tupl:
                answer = answer[i]
            return answer

        params = self.val_dist.params
        print("Inference on test examples...")
        for i in range(self.val_dist.params[0].shape[0]):
            sub_params = tuple([params[j][[i]] for j in range(len(params))])
            small_val_dist = self.val_dist.__class__(sub_params)
            val_forecast = small_val_dist.cdf(y)
            idx = []
            for k in range(len(params)):
                idx.append(torch.where(self.parameter_bins[k] > sub_params[k]))
            updated = False
            can_recalibrate = True
            for sub_list in idx:
                if len(sub_list[0]) == 0 or sub_list[0][0] == 0:
                    can_recalibrate = False
                    break

            if can_recalibrate:
                indices = [idx[k][0][0].item() - 1 for k in range(len(idx))]
                iso_reg = lookup_by_tuple(self.iso_reg_models, tuple(indices))
                res = iso_reg.predict(val_forecast)
                cdfs.append(res.flatten())
                updated = True
            if not updated:
                cdfs.append(val_forecast.flatten().numpy())

        ranking = torch.tensor(cdfs)
        dist = FlexibleDistribution((y, ranking))
        metrics = Metrics(dist, self.y_val, self.y_scale, discretization=self.n_bins)
        dic = metrics.get_metrics(decision_making=True)
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
