import torch
import numpy as np
import math
from decision_making import simulate_decision_making, DecisionMaker
from scipy.integrate import simps
import torch.distributions as D

torch.manual_seed(0)


class Metrics:
    def __init__(self, dist, y, y_scale, discretization=20):
        self.y = y.flatten()
        self.ft_yt = dist.cdf(self.y).detach().cpu()
        self.dist = dist
        self.discretization = discretization

        self.y_scale = y_scale

    def ece(self):
        ft_yt = torch.sort(self.ft_yt)[0]
        bins = np.linspace(0, 1, ft_yt.shape[0])
        res = simps(np.abs(ft_yt - bins), bins)
        return torch.tensor(res)

    def sharpness(self):
        return self.dist.dist_std.mean().detach().cpu()

    def point_calibration_errs(self, min_bin=10):
        y_sorted = torch.sort(self.y.flatten())[0]
        n_bins = self.discretization
        n_y_bins = 50
        sampled_y0 = torch.FloatTensor(n_y_bins).uniform_(
            torch.min(self.y), torch.max(self.y)
        )
        sampled_alphas = torch.linspace(0, 1, n_bins)
        right_alphas = sampled_alphas[1:].reshape(-1, 1).flatten()
        left_alphas = sampled_alphas[:-1].reshape(-1, 1).flatten()

        cdf_vals = self.ft_yt.flatten()
        total_err = torch.zeros(n_y_bins, n_bins)
        count = 0
        for k in range(n_y_bins):
            if "Composition" in self.dist.__class__.__name__:
                threshold_vals = (
                    self.dist.cdf(sampled_y0[k].to(self.y.get_device()))
                    .detach()
                    .cpu()
                    .reshape(1, -1, 1)
                )
            else:
                threshold_vals = self.dist.cdf(sampled_y0[k].view(-1, 1)).reshape(
                    1, -1, 1
                )
            selected_indices = (threshold_vals < right_alphas) & (
                threshold_vals >= left_alphas
            )  # 2 x 2100 x 199
            num_selected = selected_indices.type(torch.int).sum(dim=1)  # 2 x 199
            indices = torch.where(num_selected >= min_bin)
            total_err[k][indices[0]] = -1.0
            for x in range(len(indices[0])):
                i = indices[0][x]
                j = indices[1][x]
                selected_cdf = cdf_vals[selected_indices[i, :, j]]
                diff_from_uniform = torch.abs(
                    torch.sort(selected_cdf)[0]
                    - torch.linspace(0.0, 1.0, selected_cdf.shape[0])
                ).mean()  # *selected_cdf.shape[0]/cdf_vals.shape[0]
                total_err[k, x] = diff_from_uniform
        return total_err, sampled_y0, sampled_alphas.repeat(n_y_bins, 1)

    def point_calibration_error(self, min_bin=10):
        y_sorted = torch.sort(self.y.flatten())[0]
        n_bins = self.discretization
        n_y_bins = 50
        sampled_y0 = torch.FloatTensor(n_y_bins).uniform_(
            torch.min(self.y), torch.max(self.y)
        )
        sampled_alphas = torch.linspace(0, 1, n_bins)
        right_alphas = sampled_alphas[1:].reshape(-1, 1).flatten()
        left_alphas = sampled_alphas[:-1].reshape(-1, 1).flatten()

        cdf_vals = self.ft_yt.flatten()
        total_err = 0.0
        count = 0
        for k in range(n_y_bins):
            if "Composition" in self.dist.__class__.__name__:
                threshold_vals = (
                    self.dist.cdf(sampled_y0[k].to(self.y.get_device()))
                    .detach()
                    .cpu()
                    .reshape(1, -1, 1)
                )
            else:
                threshold_vals = self.dist.cdf(sampled_y0[k].view(-1, 1)).reshape(
                    1, -1, 1
                )
            selected_indices = (threshold_vals < right_alphas) & (
                threshold_vals >= left_alphas
            )  # 2 x 2100 x 199
            num_selected = selected_indices.type(torch.int).sum(dim=1)  # 2 x 199
            indices = torch.where(num_selected >= min_bin)
            errs = torch.zeros(len(indices[0]))
            for x in range(len(indices[0])):
                i = indices[0][x]
                j = indices[1][x]
                selected_cdf = cdf_vals[selected_indices[i, :, j]]
                diff_from_uniform = torch.abs(
                    torch.sort(selected_cdf)[0]
                    - torch.linspace(0.0, 1.0, selected_cdf.shape[0])
                ).mean()  # *selected_cdf.shape[0]/cdf_vals.shape[0]
                errs[x] = diff_from_uniform
                count += 1
            total_err += errs.sum()
        return total_err / count

    def distribution_calibration_error(self):
        second_mom = self.dist.second_mom()
        first_mom = self.dist.mean()
        cdf_vals = self.ft_yt.flatten()

        #        n = self.discretization
        n = 20
        parameter_bins = [
            torch.linspace(torch.min(first_mom), torch.max(first_mom), n + 1),
            torch.linspace(torch.min(second_mom), torch.max(second_mom), n + 1),
        ]

        dist_cal_error = torch.Tensor([0.0])
        count = 0.0
        for i in range(n):
            for j in range(n):
                condition = (
                    (first_mom <= parameter_bins[0][i + 1])
                    & (first_mom > parameter_bins[0][i])
                    & (second_mom <= parameter_bins[1][j + 1])
                    & (second_mom > parameter_bins[1][j])
                )
                indices = torch.where(condition)
                selected_cdf = cdf_vals[indices[0]]
                if len(selected_cdf) > 0:
                    dist_cal_error += torch.abs(
                        torch.sort(selected_cdf)[0]
                        - torch.linspace(0.0, 1.0, selected_cdf.shape[0])
                    ).mean()  # *selected_cdf.shape[0]/cdf_vals.shape[0]
                    count += 1
        return dist_cal_error / (count + 1e-5)

    def point_calibration_error_uniform_mass(self):
        n_bins = self.discretization
        n_y_bins = 50
        #        labels_sorted = self.y.flatten().sort()[0]
        #        thresholds = labels_sorted[((torch.rand(50) * 0.8 + 0.1) * self.y.shape[0]).type(torch.long)].detach().cpu()
        thresholds = torch.linspace(self.y.min(), self.y.max(), 50)

        count = 0
        pce_mean = 0
        bin_size = int(self.y.shape[0] / self.discretization)
        cdf_vals = self.ft_yt.flatten()
        for i in range(n_y_bins):
            if "Composition" in self.dist.__class__.__name__:
                threshold_vals = self.dist.cdf(
                    thresholds[[i]].to(self.y.get_device())
                ).flatten()
            else:
                threshold_vals = self.dist.cdf(thresholds[i].view(-1, 1)).flatten()

            sorted_thresholds, sorted_indices = torch.sort(threshold_vals)
            for x in range(self.discretization):
                if x != self.discretization - 1:
                    selected_indices = sorted_indices[x * bin_size : (x + 1) * bin_size]
                else:
                    selected_indices = sorted_indices[(x) * bin_size :]
                selected_cdf = cdf_vals[selected_indices]
                pce_mean += torch.abs(
                    torch.sort(selected_cdf)[0]
                    - torch.linspace(0.0, 1.0, selected_cdf.shape[0])
                ).mean()  # *selected_cdf.shape[0]/cdf_vals.shape[0]
                count += 1
        return pce_mean / count

    def point_calibration_error_uniform_mass_errs(self):
        n_bins = self.discretization
        n_y_bins = 50
        with torch.no_grad():
            labels_sorted = torch.sort(self.y.flatten())[0]
            thresholds = torch.linspace(
                labels_sorted[int(0.2 * len(self.y))],
                labels_sorted[int(0.8 * len(self.y))],
                50,
            )
            vals = []
            for k in range(n_y_bins):
                sub = self.dist.cdf(thresholds[[k]]).unsqueeze(dim=0)
                vals.append(sub)
            threshold_vals = torch.cat(vals, dim=0)
            sorted_thresholds, sorted_indices = torch.sort(threshold_vals, dim=1)
        total = 0
        count = 0
        pce_mean = 0
        cdf_vals = self.dist.cdf(self.y.flatten())
        bin_size = int(cdf_vals.shape[0] / n_bins)
        errs = torch.zeros(n_y_bins, n_bins)
        all_subgroups = torch.split(sorted_indices, bin_size, dim=1)
        alphas = torch.zeros(n_y_bins, n_bins + 1)
        all_sorted_thresh = torch.split(sorted_thresholds, bin_size, dim=1)

        for i in range(n_y_bins):
            for j in range(n_bins):
                alphas[i, j] = all_sorted_thresh[j][i][0]
            alphas[i, -1] = all_sorted_thresh[-1][i][-1]

        if cdf_vals.shape[0] % n_bins == 0:
            for i, selected_indices in enumerate(all_subgroups):
                selected_cdf = cdf_vals[selected_indices]
                diff_from_uniform = torch.abs(
                    torch.sort(selected_cdf)[0]
                    - torch.linspace(0.0, 1.0, selected_cdf.shape[1])
                ).mean(dim=1)
                errs[:, i] = diff_from_uniform
        else:
            remove_last = all_subgroups[: n_bins - 1]
            for i, selected_indices in enumerate(remove_last):
                selected_cdf = cdf_vals[selected_indices]
                diff_from_uniform = torch.abs(
                    torch.sort(selected_cdf)[0]
                    - torch.linspace(0.0, 1.0, selected_cdf.shape[1])
                ).mean(dim=1)
                errs[:, i] = diff_from_uniform
            last = torch.hstack(all_subgroups[n_bins - 1 :])
            selected_cdf = cdf_vals[last]
            diff_from_uniform = torch.abs(
                torch.sort(selected_cdf)[0]
                - torch.linspace(0.0, 1.0, selected_cdf.shape[1])
            ).mean(dim=1)
            errs[:, -1] = diff_from_uniform
        #            print(errs)

        return errs, thresholds, alphas

    def decision_unbiasedness(self):
        n_bins = self.discretization
        n_y_bins = 50
        with torch.no_grad():
            labels_sorted = torch.sort(self.y.flatten())[0]
            thresholds = torch.linspace(
                labels_sorted[int(0.2 * len(self.y))],
                labels_sorted[int(0.8 * len(self.y))],
                50,
            )

            vals = []
            for k in range(n_y_bins):
                sub = self.dist.cdf(thresholds[[k]]).unsqueeze(dim=0)
                vals.append(sub)
            threshold_vals = torch.cat(vals, dim=0)
            sorted_thresholds, sorted_indices = torch.sort(threshold_vals, dim=1)
        cdf_vals = self.dist.cdf(self.y.flatten())
        errs = torch.zeros(n_y_bins, n_bins)
        alphas = torch.linspace(0.05, 0.95, self.discretization)
        for i, a in enumerate(alphas):
            for j, t in enumerate(thresholds):
                selected_cdf = cdf_vals[torch.where(threshold_vals[j] <= a)]
                selected_cdf_2 = cdf_vals[torch.where(threshold_vals[j] > a)]
                if len(selected_cdf) > 200 and len(selected_cdf_2) > 200:
                    diff_from_uniform = (
                        torch.abs(
                            torch.sort(selected_cdf)[0]
                            - torch.linspace(0.0, 1.0, selected_cdf.shape[0])
                        ).mean()
                        + torch.abs(
                            torch.sort(selected_cdf_2)[0]
                            - torch.linspace(0.0, 1.0, selected_cdf_2.shape[0])
                        ).mean()
                    )
                    errs[j, i] = diff_from_uniform

        return errs, thresholds, alphas.repeat(n_y_bins, 1)

    #        idx = (errs==torch.max(errs)).nonzero()[0]
    #        return errs, torch.Tensor([thresholds[idx[0]]]), alphas[idx[1]], alphas[idx[1]] + alphas[1] - alphas[0]

    def threshold_calibration_error(self):
        n_bins = self.discretization
        n_y_bins = 50
        with torch.no_grad():
            labels_sorted = torch.sort(self.y.flatten())[0]
            thresholds = torch.linspace(
                labels_sorted[int(0.2 * len(self.y))],
                labels_sorted[int(0.8 * len(self.y))],
                50,
            )

            vals = []
            for k in range(n_y_bins):
                sub = self.dist.cdf(thresholds[[k]]).unsqueeze(dim=0)
                vals.append(sub)
            threshold_vals = torch.cat(vals, dim=0)
            sorted_thresholds, sorted_indices = torch.sort(threshold_vals, dim=1)
        cdf_vals = self.dist.cdf(self.y.flatten())
        errs1 = torch.zeros(n_y_bins, n_bins)
        errs2 = torch.zeros(n_y_bins, n_bins)

        alphas = torch.linspace(0.05, 0.95, self.discretization)
        for i, a in enumerate(alphas):
            for j, t in enumerate(thresholds):
                selected_cdf = cdf_vals[torch.where(threshold_vals[j] <= a)]
                selected_cdf_2 = cdf_vals[torch.where(threshold_vals[j] > a)]
                if len(selected_cdf) > 20 and len(selected_cdf_2) > 20:
                    diff_from_uniform1 = torch.abs(
                        torch.sort(selected_cdf)[0]
                        - torch.linspace(0.0, 1.0, selected_cdf.shape[0])
                    ).mean()
                    errs1[j, i] = diff_from_uniform1
                    diff_from_uniform2 = torch.abs(
                        torch.sort(selected_cdf_2)[0]
                        - torch.linspace(0.0, 1.0, selected_cdf_2.shape[0])
                    ).mean()
                    errs2[j, i] = diff_from_uniform2

        idx = torch.where(errs1 != 0)
        return (
            torch.mean(errs1[idx]),
            torch.mean(errs2[idx]),
            torch.mean(errs1[idx] + errs2[idx]),
        )

    def threshold_calibration_error_all(self):
        n_bins = self.discretization
        n_y_bins = 50
        with torch.no_grad():
            labels_sorted = torch.sort(self.y.flatten())[0]
            thresholds = torch.linspace(
                labels_sorted[0], min(labels_sorted[-1], 10), 50
            )

            vals = []
            for k in range(n_y_bins):
                sub = self.dist.cdf(thresholds[[k]]).unsqueeze(dim=0)
                vals.append(sub)
            threshold_vals = torch.cat(vals, dim=0)
            sorted_thresholds, sorted_indices = torch.sort(threshold_vals, dim=1)
        cdf_vals = self.dist.cdf(self.y.flatten())
        errs1 = torch.zeros(n_y_bins, n_bins)
        errs2 = torch.zeros(n_y_bins, n_bins)

        alphas = torch.linspace(0.05, 0.95, self.discretization)
        for i, a in enumerate(alphas):
            for j, t in enumerate(thresholds):
                selected_cdf = cdf_vals[torch.where(threshold_vals[j] <= a)]
                selected_cdf_2 = cdf_vals[torch.where(threshold_vals[j] > a)]
                if len(selected_cdf) > 20 and len(selected_cdf_2) > 20:
                    diff_from_uniform1 = torch.abs(
                        torch.sort(selected_cdf)[0]
                        - torch.linspace(0.0, 1.0, selected_cdf.shape[0])
                    ).mean()
                    errs1[j, i] = diff_from_uniform1
                    diff_from_uniform2 = torch.abs(
                        torch.sort(selected_cdf_2)[0]
                        - torch.linspace(0.0, 1.0, selected_cdf_2.shape[0])
                    ).mean()
                    errs2[j, i] = diff_from_uniform2

        idx = torch.where(errs1 != 0)
        return torch.mean(errs1[idx] + errs2[idx])

    def rmse(self):
        mse_loss = torch.nn.MSELoss(reduction="mean")
        rmse = torch.sqrt(mse_loss(self.y, self.dist.dist_mean())) * self.y_scale
        return rmse

    def decision_loss(self):
        #        labels_sorted = self.y.flatten().sort()[0]
        #        sampled_y0 = labels_sorted[((torch.rand(50) * 0.8 + 0.1) * self.y.shape[0]).type(torch.long)].detach().cpu()
        sampled_y0 = torch.linspace(self.y.min(), min(10, self.y.max()), 50)
        sampled_alpha = torch.linspace(0.05, 0.95, 50)
        decision_makers = []
        for i in range(len(sampled_alpha)):
            for j in range(len(sampled_y0)):
                decision_makers.append(
                    DecisionMaker(sampled_alpha[i], sampled_y0[j], self.dist)
                )

        loss = simulate_decision_making(
            decision_makers, self.dist, self.y.flatten()
        )  # self.point_recal, self.point_recal_params)
        return loss

    def get_metrics(self, decision_making=False):
        uniform_mass_pce = self.point_calibration_error_uniform_mass()

        less, greater, both = self.threshold_calibration_error()
        if decision_making:
            (
                decision_err,
                decision_loss,
                all_err,
                all_loss,
                all_y0,
                all_c,
            ) = self.decision_loss()
        else:
            decision_loss = torch.tensor(0.0)
            decision_err = torch.tensor(0.0)
            all_err = torch.tensor([0.0])
            all_loss = torch.tensor([0.0])
            all_y0 = torch.tensor([0.0])
            all_c = torch.tensor([0.0])
        return {
            "ece": self.ece(),
            "point_calibration_error_uniform_mass": uniform_mass_pce,
            "point_calibration_error": self.point_calibration_error(),
            "threshold_calibration_error_less": less,
            "threshold_calibration_error_greater": greater,
            "threshold_calibration_error_both": both,
            "threshold_calibration_error_all": self.threshold_calibration_error_all(),
            #            "distribution_calibration_error": self.distribution_calibration_error(),
            #                "rmse": self.rmse(),
            "true_vs_pred_loss": decision_err,
            "decision_loss": decision_loss,
            "all_err": all_err,
            "all_loss": all_loss,
            "all_y0": all_y0,
            "all_c": all_c,
        }
