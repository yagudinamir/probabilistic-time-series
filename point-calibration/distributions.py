from numpy import dtype
import torch
import torch.distributions as D
from scipy.interpolate import interp1d
from scipy.integrate import simps


class GaussianLaplaceMixtureDistribution:
    def __init__(self, params):
        self.params = params
        mu, var, loc, scale, self.weight = params
        sigma = torch.sqrt(var)
        self.gaussian_comp = D.Normal(mu, sigma)
        self.laplace_comp = D.Laplace(loc, scale)
        self.dist_mean = self.mean()
        self.dist_std = self.std()

    def cdf(self, y):
        gaussian_cdf = self.gaussian_comp.cdf(y) * self.weight
        laplace_cdf = self.laplace_comp.cdf(y) * (1 - self.weight)
        return laplace_cdf + gaussian_cdf

    def mean(self):
        return self.weight * self.gaussian_comp.mean + (1 - self.weight) * self.laplace_comp.mean

    def second_mom(self):
        return self.std() ** 2 + self.mean() ** 2

    def std(self):
        gaussian_part = self.weight * (torch.pow(self.gaussian_comp.mean, 2) + self.gaussian_comp.variance)
        laplace_part = (1 - self.weight) * (torch.pow(self.laplace_comp.mean, 2) + self.laplace_comp.variance)
        total = gaussian_part + laplace_part - torch.pow(self.dist_mean, 2)
        return torch.sqrt(total)

    def to(self, device):
        self.gaussian_comp = D.Normal(self.gaussian_comp.mean.to(device), self.gaussian_comp.scale.to(device))
        self.laplace_comp = D.Laplace(self.laplace_comp.mean.to(device), self.laplace_comp.scale.to(device))
        self.dist_mean = self.dist_mean.to(device)
        self.dist_std = self.dist_std.to(device)
        self.weight = self.weight.to(device)
        return self

    def detach(self):
        self.gaussian_comp = D.Normal(
            self.gaussian_comp.loc.detach().cpu(),
            self.gaussian_comp.scale.detach().cpu(),
        )
        self.laplace_comp = D.Laplace(self.laplace_comp.loc.detach().cpu(), self.laplace_comp.scale.detach().cpu())
        self.dist_mean = self.dist_mean.detach().cpu()
        self.dist_std = self.dist_std.detach().cpu()
        self.weight = self.weight.detach().cpu()
        return self


class GaussianDistribution:
    def __init__(self, params):
        self.params = params
        mu, var = params
        sigma = torch.sqrt(var)
        self.gaussian_comp = D.Normal(mu, sigma)
        self.dist_mean = self.mean()
        self.dist_std = self.std()

    def cdf(self, y):
        return self.gaussian_comp.cdf(y)

    def mean(self):
        return self.gaussian_comp.mean

    def std(self):
        return self.gaussian_comp.scale

    def second_mom(self):
        return self.std() ** 2 + self.mean() ** 2

    def to(self, device):
        self.gaussian_comp = D.Normal(self.gaussian_comp.mean.to(device), self.gaussian_comp.scale.to(device))
        self.dist_mean = self.dist_mean.to(device)
        self.dist_std = self.dist_std.to(device)
        return self

    def detach(self):
        self.gaussian_comp = D.Normal(
            self.gaussian_comp.loc.detach().cpu(),
            self.gaussian_comp.scale.detach().cpu(),
        )
        self.dist_mean = self.dist_mean.detach().cpu()
        self.dist_std = self.dist_std.detach().cpu()
        return self


class FlexibleDistribution:
    def __init__(self, params):
        self.xs, self.cdfs = params
        cdf_functions = []
        for cdf in self.cdfs:
            cdf_functions.append(interp1d(self.xs, cdf))
        self.cdf_functions = cdf_functions
        self.dist_mean = self.mean()
        self.dist_std = self.std()

    def cdf(self, y):
        y = torch.clamp(y, min=-10, max=10)
        results = []
        if len(torch.tensor(y.shape)) == 1 and y.shape[0] == len(self.cdf_functions):
            for i, f in enumerate(self.cdf_functions):
                results.append(f(y[i]).item())
            results = torch.tensor(results)
        elif len(torch.tensor(y.shape)) == 1 and y.shape[0] == 1:
            idx = torch.where(self.xs >= y[0])[0][0]
            idx_before = idx - 1
            results = self.cdfs[:, idx]
        #            for i, f in enumerate(self.cdf_functions):
        #                results.append(f(y[0]).item())
        #            results = torch.tensor(results)
        else:
            for i, f in enumerate(self.cdf_functions):
                results.append(f(y))
        #            results = torch.tensor(results).T[0, :, :]
        if results[0].dtype is torch.float64:
            results = [res.type(torch.float32) for res in results]
        results = torch.Tensor(results)
        return results

    def mean(self):
        first_mom = []
        idx = int(self.xs.shape[0] / 2)
        for cdf in self.cdfs:
            l = -simps(cdf[:idx], self.xs[:idx])
            r = simps(1 - cdf[idx:], self.xs[idx:])
            first_mom.append(l + r)
        return torch.tensor(first_mom)

    def second_mom(self):
        second_mom = []
        idx = int(self.xs.shape[0] / 2)

        for cdf in self.cdfs:
            left = -simps(self.xs[:idx] * cdf[:idx], self.xs[:idx])
            right = simps(self.xs[idx:] * (1 - cdf[idx:]), self.xs[idx:])
            second_mom.append((left + right) * 2)
        second_mom = torch.tensor(second_mom)
        return second_mom

    def std(self):
        second_mom = []
        idx = int(self.xs.shape[0] / 2)

        for cdf in self.cdfs:
            left = -simps(self.xs[:idx] * cdf[:idx], self.xs[:idx])
            right = simps(self.xs[idx:] * (1 - cdf[idx:]), self.xs[idx:])
            second_mom.append((left + right) * 2)
        second_mom = torch.tensor(second_mom)
        first_mom = self.mean()
        var = second_mom - torch.pow(first_mom, 2)
        return torch.sqrt(var)
