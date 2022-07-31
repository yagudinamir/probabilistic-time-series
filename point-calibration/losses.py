import torch
import torch.distributions as D
import math


class GaussianNLL:
    def __init__(self):
        self.name = "gaussian_nll"

    def __call__(self, y, mu, var):
        nll = 0
        sigma = torch.sqrt(var)
        comp = D.Normal(mu, sigma)
        log_prob = comp.log_prob(y)
        nll += -log_prob
        return torch.mean(nll)


class GaussianLaplaceMixtureNLL:
    def __init__(self):
        self.name = "gaussian_laplace_mixture_nll"

    def __call__(self, y, mu, var, loc, scale, weight):
        gaussian_likelihood = (
            (1 / torch.sqrt(2 * math.pi * var))
            * torch.exp(-0.5 * torch.pow(y - mu, 2) / var)
            * weight
        )
        laplace_likelihood = (
            (1 / (2 * scale)) * torch.exp(-torch.abs(y - loc) / scale) * (1 - weight)
        )
        likelihood = laplace_likelihood + gaussian_likelihood
        likelihood = likelihood.clamp(min=1e-20)
        nll = -torch.log(likelihood)
        return torch.mean(nll)


class PointCalibrationLoss:
    def __init__(self, discretization, y=None):
        self.name = "pointwise_calibration_loss"
        self.discretization = discretization
        if y:
            self.labels_sorted = torch.sort(y.flatten())[0]
        else:
            self.labels_sorted = None

    def __call__(self, y, dist):
        n_bins = self.discretization
        n_y_bins = 50
        with torch.no_grad():
            if not self.labels_sorted:
                labels_sorted = torch.sort(y.flatten())[0]
            else:
                labels_sorted = self.labels_sorted
            sampled_index = ((torch.rand(n_y_bins) * 0.8 + 0.1) * y.shape[0]).type(
                torch.long
            )
            thresholds = labels_sorted[sampled_index]
            vals = []
            for k in range(n_y_bins):
                sub = dist.cdf(thresholds[k]).unsqueeze(dim=0)
                vals.append(sub)
            threshold_vals = torch.cat(vals, dim=0)
            sorted_thresholds, sorted_indices = torch.sort(threshold_vals, dim=1)
        total = 0
        count = 0
        pce_mean = 0
        cdf_vals = dist.cdf(y.flatten())
        bin_size = int(cdf_vals.shape[0] / n_bins)
        errs = torch.zeros(n_y_bins, n_bins).to(y.get_device())
        all_subgroups = torch.split(sorted_indices, bin_size, dim=1)
        if cdf_vals.shape[0] % n_bins == 0:
            for i, selected_indices in enumerate(all_subgroups):
                selected_cdf = cdf_vals[selected_indices]
                diff_from_uniform = torch.abs(
                    torch.sort(selected_cdf)[0]
                    - torch.linspace(0.0, 1.0, selected_cdf.shape[1]).to(y.get_device())
                ).mean(dim=1)
                errs[:, i] = diff_from_uniform
        else:
            remove_last = all_subgroups[: -(len(all_subgroups) - n_bins)]
            for i, selected_indices in enumerate(remove_last):
                selected_cdf = cdf_vals[selected_indices]
                diff_from_uniform = torch.abs(
                    torch.sort(selected_cdf)[0]
                    - torch.linspace(0.0, 1.0, selected_cdf.shape[1]).to(y.get_device())
                ).mean(dim=1)
                errs[:, i] = diff_from_uniform
            last = torch.hstack(all_subgroups[-(len(all_subgroups) - n_bins) :])
            selected_cdf = cdf_vals[last]
            diff_from_uniform = torch.abs(
                torch.sort(selected_cdf)[0]
                - torch.linspace(0.0, 1.0, selected_cdf.shape[1]).to(y.get_device())
            ).mean(dim=1)
            errs[:, -1] = diff_from_uniform

        return torch.mean(errs)


class CalibrationLoss:
    def __init__(self):
        self.name = "calibration_loss"

    def __call__(self, y, dist):
        cdf_vals = dist.cdf(y.flatten())
        calibration_error = torch.abs(
            torch.sort(cdf_vals)[0]
            - torch.linspace(0.0, 1.0, cdf_vals.shape[0]).to(y.get_device())
        ).mean()
        return calibration_error
