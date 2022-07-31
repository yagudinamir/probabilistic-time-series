import torch

RANGE = [-10, 10]
RESOLUTION = 8000
from scipy.integrate import simps


class CompositionDist:
    def __init__(self, r, f):
        self.r = r
        self.f = f

    def cdf(self, y):
        inner = self.f.cdf(y)
        info = torch.cat(
            tuple([param.unsqueeze(dim=0) for param in self.f.params]), dim=0
        )
        #        info = torch.cat((self.f.mean.unsqueeze(dim=0), self.f.scale.unsqueeze(dim=0)), dim=0)
        out = self.r.cdf(inner, info)
        assert inner.shape[-1] == info.shape[-1]
        return out

    def dist_mean(self):
        y = torch.linspace(-10, 10, 8000).to(r.get_device())
        out = self.cdf(y).detach().cpu().numpy()
        first_mom = []
        idx = int(self.xs.shape[0] / 2)
        for cdf in out:
            l = -simps(cdf[:idx], self.xs[:idx])
            r = simps(1 - cdf[idx:], self.xs[idx:])
            first_mom.append(l + r)
        return torch.tensor(first_mom)

    def to(self, device):
        self.r = self.r.to(device)
        self.f = self.f.to(device)
        return self

    def detach(self):
        self.r = self.r.cpu()
        self.f = self.f.detach()
        return self


class RecalibrationLayer:
    def __init__(self, outer_layer, outer_alphas, threshold):
        self.outer_layer = outer_layer
        self.outer_alphas = torch.Tensor(outer_alphas)
        self.threshold = threshold

    def cdf(self, inner, y):
        quantile_cdfs = inner.cdf(self.threshold)
        quantile_sorted, sorted_indices = torch.sort(quantile_cdfs)

        current = inner.cdf(y)
        indices = torch.searchsorted(self.outer_alphas, quantile_sorted)

        if y.shape[0] == 1 and y.shape[1] == 1:
            out = torch.zeros(current.shape)
            for j in torch.unique(indices):
                placeholder = indices == j
                y_vals = current[sorted_indices[placeholder]].detach().cpu().numpy()
                if j < len(self.outer_layer):
                    out[sorted_indices[placeholder]] = torch.tensor(
                        self.outer_layer[j].predict(y_vals)
                    )
                else:
                    out[sorted_indices[placeholder]] = torch.tensor(y_vals)
        elif inner.mean().shape[0] == y.shape[0]:
            out = torch.zeros(current.shape)
            for j in torch.unique(indices):
                placeholder = indices == j
                y_vals = current[sorted_indices[placeholder]].detach().cpu().numpy()
                if j < len(self.outer_layer):
                    out[sorted_indices[placeholder]] = torch.tensor(
                        self.outer_layer[j].predict(y_vals)
                    )
                else:
                    out[sorted_indices[placeholder]] = torch.tensor(y_vals)
        else:
            if inner.mean().shape[0] == 1:
                j = indices[0]
                if j < len(self.outer_layer):
                    out = torch.tensor(
                        self.outer_layer[j].predict(
                            current.detach().cpu().numpy().flatten()
                        )
                    )
                else:
                    out = torch.tensor(current.detach().cpu().numpy().flatten())

        return out.numpy()


class AlphaRecalibrationLayer:
    def __init__(self, outer_layer, outer_alpha, threshold):
        self.outer_layer = outer_layer
        self.outer_alphas = torch.Tensor([outer_alpha])
        self.threshold = threshold

    def cdf(self, inner, y):

        quantile_cdfs = inner.cdf(self.threshold)
        quantile_sorted, sorted_indices = torch.sort(quantile_cdfs)
        current = inner.cdf(y)
        indices = torch.searchsorted(self.outer_alphas, quantile_sorted)

        if y.shape[0] == 1 and y.shape[1] == 1:
            out = torch.zeros(current.shape)
            for j in torch.unique(indices):
                placeholder = indices == j
                y_vals = current[sorted_indices[placeholder]].detach().cpu().numpy()
                if j <= 1:
                    out[sorted_indices[placeholder]] = torch.tensor(
                        self.outer_layer[j].predict(y_vals)
                    )
                else:
                    out[sorted_indices[placeholder]] = torch.tensor(y_vals)
        elif inner.mean().shape[0] == y.shape[0]:
            out = torch.zeros(current.shape)
            for j in torch.unique(indices):
                placeholder = indices == j
                y_vals = current[sorted_indices[placeholder]].detach().cpu().numpy()
                if j <= 1:
                    out[sorted_indices[placeholder]] = torch.tensor(
                        self.outer_layer[j].predict(y_vals)
                    )
                else:
                    out[sorted_indices[placeholder]] = torch.tensor(y_vals)
        else:
            if inner.mean().shape[0] == 1:
                j = indices[0]
                if j <= 1:
                    out = torch.tensor(
                        self.outer_layer[j].predict(
                            current.detach().cpu().numpy().flatten()
                        )
                    )
                else:
                    out = torch.tensor(current.detach().cpu().numpy().flatten())

        return out.numpy()
