import torch
import torch.nn as nn


class SimpleSigmoidLayer(torch.nn.Module):
    def __init__(self, n_dim, no_in=False):
        super(SimpleSigmoidLayer, self).__init__()
        #        self.log_a = nn.Parameter(torch.randn(n_dim, 1))
        #        self.log_a = torch.ones(n_dim, 1)
        self.no_in = no_in
        if self.no_in:
            self.a = torch.zeros(n_dim, 1)
        else:
            self.log_a = nn.Parameter(torch.randn(n_dim, 1))
        self.b = nn.Parameter(torch.randn(n_dim, 1))
        self.unnormalized_w = nn.Parameter(torch.randn(1, n_dim))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        #        a = torch.exp(self.log_a)
        if self.no_in:
            a = self.a.to(x.get_device())
        else:
            a = torch.exp(self.log_a)
        inner = a * x + self.b
        out = torch.sigmoid(inner)
        w = self.softmax(self.unnormalized_w)
        xnew = torch.matmul(w, out)
        return xnew


class OneLayerFlow(torch.nn.Module):
    def __init__(self, in_dim, n_dim):
        super(OneLayerFlow, self).__init__()
        self.log_a = nn.Parameter(torch.randn(n_dim, in_dim))
        self.b = nn.Parameter(torch.randn(n_dim, 1))
        self.unnormalized_w = nn.Parameter(torch.randn(1, n_dim))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        #        a = torch.exp(self.log_a)
        a = torch.exp(self.log_a)
        inner = torch.matmul(a, x) + self.b
        out = torch.sigmoid(inner)
        w = self.softmax(self.unnormalized_w)
        xnew = torch.matmul(w, out)
        return xnew


class SigmoidLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SigmoidLayer, self).__init__()
        self.unnormalized_w = nn.Parameter(torch.randn(out_dim, out_dim))
        self.log_a = nn.Parameter(torch.randn(out_dim, 1))
        self.b = nn.Parameter(torch.randn(out_dim, 1))
        self.unnormalized_u = nn.Parameter(torch.randn(out_dim, in_dim))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):

        a = torch.exp(self.log_a)
        u = self.softmax(self.unnormalized_u)
        w = self.softmax(self.unnormalized_w)
        inner = torch.matmul((a * u), x) + self.b
        out = torch.sigmoid(inner)
        xnew = torch.matmul(w, out)
        return xnew


class DenseSigmoidFlow(torch.nn.Module):
    def __init__(self, num_layers, n_dim):
        super(DenseSigmoidFlow, self).__init__()
        self.num_layers = num_layers
        self.n_dim = n_dim
        l = [SigmoidLayer(in_dim=1, out_dim=n_dim)]
        for i in range(num_layers - 2):
            l.append(SigmoidLayer(in_dim=n_dim, out_dim=n_dim))
        l.append(SigmoidLayer(in_dim=n_dim, out_dim=1))
        self.layers = nn.ModuleList(l)

    def forward(self, x):
        x = x.reshape(-1, 1, 1)
        for layer in self.layers:
            x = layer(x)
        return x

    def cdf(self, x):
        bottom = self.forward(torch.Tensor([0]).to(x.get_device()))
        top = self.forward(torch.Tensor([1]).to(x.get_device()))

        in_shape = x.shape
        if len(in_shape) == 2:
            res = self.forward(x.flatten())
            final = res.reshape(in_shape)
        else:
            final = self.forward(x).flatten()
        out_shape = final.shape
        final = (final - bottom.flatten()) / (top.flatten() - bottom.flatten())
        assert in_shape == out_shape
        return final


class SigmoidFlow(torch.nn.Module):
    def __init__(self, num_layers, n_dim):
        super(SigmoidFlow, self).__init__()
        self.num_layers = num_layers
        self.n_dim = n_dim
        l = [SimpleSigmoidLayer(n_dim=n_dim, no_in=False)]

    #        for i in range(self.num_layers-1):
    #            l.append(SimpleSigmoidLayer(n_dim=n_dim, no_in=False))
    #        l.append(torch.nn.Sigmoid())
    #        self.layers =  nn.ModuleList(l)

    def forward(self, x):
        x = x.reshape(-1, 1, 1)
        for layer in self.layers:
            x = layer(x)
        return x

    def cdf(self, x):
        bottom = self.forward(torch.Tensor([0]).to(x.get_device()))
        top = self.forward(torch.Tensor([1]).to(x.get_device()))

        in_shape = x.shape
        if len(in_shape) == 2:
            res = self.forward(x.flatten())
            final = res.reshape(in_shape)
        else:
            final = self.forward(x).flatten()
        out_shape = final.shape
        final = (final - bottom.flatten()) / (top.flatten() - bottom.flatten())
        assert in_shape == out_shape
        return final


class SigmoidFlow2D(nn.Module):
    def __init__(self, num_layers, n_dim):
        super(SigmoidFlow2D, self).__init__()
        self.num_layers = num_layers
        self.n_dim = n_dim
        l = [SigmoidLayer(in_dim=2, out_dim=n_dim)]
        for i in range(num_layers - 2):
            l.append(SigmoidLayer(in_dim=n_dim, out_dim=n_dim))
        l.append(SigmoidLayer(in_dim=n_dim, out_dim=1))
        self.layers = nn.ModuleList(l)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1),
        )

    def forward(self, x, fy0):
        fy0 = fy0.reshape(-1, 1, 1)
        x = x.reshape(-1, 1, 1)
        theta = self.mlp(fy0)
        mono_input = torch.hstack((x, theta))
        for layer in self.layers:
            mono_input = layer(mono_input)
        return mono_input

    def cdf(self, x, fy0):
        assert x.shape == fy0.shape
        in_shape = x.shape
        device = x.get_device()
        if len(in_shape) == 2:
            res = self.forward(x.flatten(), fy0.flatten())
            final = res.reshape(in_shape)
            bottom = self.forward(
                torch.zeros(x.shape).to(device).flatten(), fy0.flatten()
            ).reshape(in_shape)
            top = self.forward(
                torch.ones(x.shape).to(device).flatten(), fy0.flatten()
            ).reshape(in_shape)
        else:
            final = self.forward(x, fy0).flatten()
            bottom = self.forward(torch.zeros(x.shape).to(device), fy0).flatten()
            top = self.forward(torch.ones(x.shape).to(device), fy0).flatten()
        if torch.sum(torch.isnan(1 / (top - bottom))) > 0:
            import pdb

            pdb.set_trace()
        final = (final - bottom) / (top - bottom)
        out_shape = final.shape
        assert in_shape == out_shape
        return final


class SigmoidFlowND(nn.Module):
    def __init__(self, n_in, num_layers, n_dim):
        super(SigmoidFlowND, self).__init__()
        #            torch.set_default_tensor_type(torch.DoubleTensor)
        self.num_layers = num_layers
        self.n_dim = n_dim
        self.n_in = n_in
        l = [SigmoidLayer(in_dim=n_in, out_dim=n_dim)]
        for i in range(num_layers - 1):
            l.append(SigmoidLayer(in_dim=n_dim, out_dim=n_dim))
        l.append(SigmoidLayer(in_dim=n_dim, out_dim=1))
        self.layers = nn.ModuleList(l)

        mlps = []
        for i in range(n_in - 1):
            mlps.append(
                torch.nn.Sequential(
                    torch.nn.Linear(1, 10),
                    torch.nn.ReLU(),
                    torch.nn.Linear(10, 10),
                    torch.nn.ReLU(),
                    torch.nn.Linear(10, 1),
                )
            )
        self.mlps = nn.ModuleList(mlps)

    #            self.mlp = torch.nn.Sequential(torch.nn.Linear(n_in-1, 30), torch.nn.ReLU(),torch.nn.Linear(30, 30), torch.nn.ReLU(), torch.nn.Linear(30, n_in-1))
    #            self.mlps = nn.ModuleList(mlps)

    def forward(self, x, extra=None):
        #            import pdb
        #            pdb.set_trace()
        x = x.reshape(-1, 1, 1)

        if type(extra) == torch.Tensor:
            #                import pdb
            #                pdb.set_trace()
            extras = extra.chunk(self.n_in - 1)
            extras = [p.reshape(-1, 1, 1) for p in extras]

            all_input = [x]
            for i in range(len(extras)):
                all_input.append(self.mlps[i](extras[i]))
                mono_input = torch.cat(tuple(all_input), dim=1)
        #                res = self.mlp(extra.T.unsqueeze(1))
        #                res = res.reshape(res.shape[0], res.shape[2], res.shape[1])
        #                import pdb
        #                pdb.set_trace()
        #                mono_input = torch.cat((x, res), dim=1)
        else:
            mono_input = x
        #            y0 = y0.reshape(-1, 1, 1)
        #            fy0 = fy0.reshape(-1, 1, 1)
        for layer in self.layers:
            mono_input = layer(mono_input)
        if torch.sum(torch.isnan(mono_input)) > 0:
            import pdb

            pdb.set_trace()
        return mono_input

    def cdf(self, x, extra=None):
        if type(extra) == torch.Tensor:
            assert x.shape[-1] == extra.shape[-1]
        device = x.get_device()
        in_shape = x.shape

        final = self.forward(x, extra).flatten()
        bottom = self.forward(torch.zeros(x.shape).to(device), extra).flatten()
        top = self.forward(torch.ones(x.shape).to(device), extra).flatten()
        if torch.sum(torch.isinf(1 / (top - bottom))) > 0:
            import pdb

            pdb.set_trace()

        final = (final - bottom) / (top - bottom)
        if torch.sum(torch.isnan(final)) > 0:
            import pdb

            pdb.set_trace()

        out_shape = final.shape
        assert in_shape == out_shape
        return final


class SigmoidFlowNDSingleMLP(nn.Module):
    def __init__(self, n_in, num_layers, n_dim):
        super(SigmoidFlowNDSingleMLP, self).__init__()
        #            torch.set_default_tensor_type(torch.DoubleTensor)
        self.num_layers = num_layers
        self.n_dim = n_dim
        self.n_in = n_in
        l = [SigmoidLayer(in_dim=2, out_dim=n_dim)]
        for i in range(num_layers - 1):
            l.append(SigmoidLayer(in_dim=n_dim, out_dim=n_dim))
        l.append(SigmoidLayer(in_dim=n_dim, out_dim=1))
        self.layers = nn.ModuleList(l)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_in - 1, 50), torch.nn.ReLU(), torch.nn.Linear(50, 1)
        )

    def forward(self, x, extra=None):
        #            import pdb
        #            pdb.set_trace()
        x = x.reshape(-1, 1, 1)
        if type(extra) == torch.Tensor:
            all_input = [x]
            res = self.mlp(extra.T.unsqueeze(1))
            res = res.reshape(res.shape[0], res.shape[2], res.shape[1])
            #                import pdb
            #                pdb.set_trace()
            mono_input = torch.cat((x, res), dim=1)
        else:
            mono_input = x
        #            y0 = y0.reshape(-1, 1, 1)
        #            fy0 = fy0.reshape(-1, 1, 1)
        for layer in self.layers:
            mono_input = layer(mono_input)
        if torch.sum(torch.isnan(mono_input)) > 0:
            import pdb

            pdb.set_trace()
        return mono_input

    def cdf(self, x, extra=None):
        if type(extra) == torch.Tensor:
            assert x.shape[-1] == extra.shape[-1]
        device = x.get_device()
        in_shape = x.shape
        final = self.forward(x, extra).flatten()
        bottom = self.forward(torch.zeros(x.shape).to(device), extra).flatten()
        top = self.forward(torch.ones(x.shape).to(device), extra).flatten()
        final = (final - bottom) / (top - bottom).clamp(min=1e-10)
        if torch.sum(torch.isnan(final)) > 0:
            import pdb

            pdb.set_trace()

        out_shape = final.shape
        assert in_shape == out_shape
        return final


class SigmoidFlowNDSingleMLPDropout(nn.Module):
    def __init__(self, n_in, num_layers, n_dim):
        super(SigmoidFlowNDSingleMLPDropout, self).__init__()
        #            torch.set_default_tensor_type(torch.DoubleTensor)
        self.num_layers = num_layers
        self.n_dim = n_dim
        self.n_in = n_in
        #            l = [SigmoidLayer(in_dim=2, out_dim=n_dim)]
        #            for i in range(num_layers-1):
        #                l.append(SigmoidLayer(in_dim=n_dim, out_dim=n_dim))
        #            l.append(SigmoidLayer(in_dim=n_dim, out_dim=1))
        l = [OneLayerFlow(in_dim=2, n_dim=n_dim)]
        self.layers = nn.ModuleList(l)

        #            mlps = []
        #            for i in range(n_in-1):
        #                mlps.append(torch.nn.Sequential(torch.nn.Linear(1, 10), torch.nn.ReLU(),torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)))
        #            self.mlps = nn.ModuleList(mlps)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_in - 1, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1)
        )

    def forward(self, x, extra=None):
        #            import pdb
        #            pdb.set_trace()
        x = x.reshape(-1, 1, 1)
        if type(extra) == torch.Tensor:
            all_input = [x]
            res = self.mlp(extra.T.unsqueeze(1))
            res = res.reshape(res.shape[0], res.shape[2], res.shape[1])
            #                import pdb
            #                pdb.set_trace()
            mono_input = torch.cat((x, res), dim=1)
        else:
            mono_input = x
        #            y0 = y0.reshape(-1, 1, 1)
        #            fy0 = fy0.reshape(-1, 1, 1)
        for layer in self.layers:
            mono_input = layer(mono_input)
        if torch.sum(torch.isnan(mono_input)) > 0:
            import pdb

            pdb.set_trace()
        return mono_input

    def cdf(self, x, extra=None):
        if type(extra) == torch.Tensor:
            assert x.shape[-1] == extra.shape[-1]
        device = x.get_device()
        in_shape = x.shape
        final = self.forward(x, extra).flatten()
        bottom = self.forward(torch.zeros(x.shape).to(device), extra).flatten()
        top = self.forward(torch.ones(x.shape).to(device), extra).flatten()
        if torch.sum(torch.isinf(1 / (top - bottom))) > 0:
            import pdb

            pdb.set_trace()

        final = (final - bottom) / (top - bottom).clamp(min=1e-10)
        if torch.sum(torch.isnan(final)) > 0:
            import pdb

            pdb.set_trace()

        out_shape = final.shape
        assert in_shape == out_shape
        return final


class SigmoidFlowNDMonotonic(nn.Module):
    def __init__(self, n_in, num_layers, n_dim):
        super(SigmoidFlowND, self).__init__()
        #            torch.set_default_tensor_type(torch.DoubleTensor)
        self.num_layers = num_layers
        self.n_dim = n_dim
        self.n_in = n_in
        l = [SigmoidLayer(in_dim=n_in, out_dim=n_dim)]
        for i in range(num_layers - 1):
            l.append(SigmoidLayer(in_dim=n_dim, out_dim=n_dim))
        l.append(SigmoidLayer(in_dim=n_dim, out_dim=1))
        self.layers = nn.ModuleList(l)

    def forward(self, x, extra=None):
        #            import pdb
        #            pdb.set_trace()
        x = x.reshape(-1, 1, 1)

        if type(extra) == torch.Tensor:
            #                import pdb
            #                pdb.set_trace()
            all_input = [x]
            extras = extra.chunk(self.n_in - 1)

            for i in range(len(extras)):
                all_input.append(extras[i].reshape(-1, 1, 1))
            mono_input = torch.cat(tuple(all_input), dim=1)
        #                res = self.mlp(extra.T.unsqueeze(1))
        #                res = res.reshape(res.shape[0], res.shape[2], res.shape[1])
        #                import pdb
        #                pdb.set_trace()
        #                mono_input = torch.cat((x, res), dim=1)
        else:
            mono_input = x
        #            y0 = y0.reshape(-1, 1, 1)
        #            fy0 = fy0.reshape(-1, 1, 1)
        for layer in self.layers:
            mono_input = layer(mono_input)
        if torch.sum(torch.isnan(mono_input)) > 0:
            import pdb

            pdb.set_trace()
        return mono_input

    def cdf(self, x, extra=None):
        if type(extra) == torch.Tensor:
            assert x.shape[-1] == extra.shape[-1]
        device = x.get_device()
        in_shape = x.shape

        final = self.forward(x, extra).flatten()
        bottom = self.forward(torch.zeros(x.shape).to(device), extra).flatten()
        top = self.forward(torch.ones(x.shape).to(device), extra).flatten()
        if torch.sum(torch.isnan(1 / (top - bottom))) > 0:
            import pdb

            pdb.set_trace()

        final = (final - bottom) / (top - bottom)
        if torch.sum(torch.isnan(final)) > 0:
            import pdb

            pdb.set_trace()

        out_shape = final.shape
        assert in_shape == out_shape
        return final
