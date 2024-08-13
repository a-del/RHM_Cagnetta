import torch
from torch import nn

from optim_loss import CLAPPUnsupervisedHalfMasking


class NonOverlappingConv1d(nn.Module):
    def __init__(
        self, input_channels, out_channels, out_dim, patch_size, bias=False
    ):
        super(NonOverlappingConv1d, self).__init__()
        self.weight = nn.Parameter( # input [bs, cin, space / 2, 2], weight [cout, cin, 1, 2]
            torch.randn(
                out_channels,
                input_channels,
                1,
                patch_size,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, out_dim))   # why should there be a bias different for each location? That's not like a Conv??
        else:
            self.register_parameter("bias", None)

        self.input_channels = input_channels
        self.patch_size = patch_size

    def forward(self, x):
        bs, cin, d = x.shape
        x = x.view(bs, 1, cin, d // self.patch_size, self.patch_size) # [bs, 1, cin, space // patch_size, patch_size]
        x = x * self.weight # [bs, cout, cin, space // patch_size, patch_size]
        x = x.sum(dim=[-1, -3]) # [bs, cout, space // patch_size]
        x /= self.input_channels ** .5
        if self.bias is not None:
            x += self.bias * 0.1
        return x


class NonOverlappingConv1dReLU(NonOverlappingConv1d):
    def __init__(
        self, input_channels, out_channels, out_dim, patch_size, bias=False
    ):
        super(NonOverlappingConv1dReLU, self).__init__(input_channels, out_channels, out_dim, patch_size, bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = super(NonOverlappingConv1dReLU, self).forward(x)
        return self.relu(x)


class CNN(nn.Module):
    def __init__(self, input_channels, h, out_dim, num_layers, patch_size=2, bias=False, layerwise=False,
                 last_lin_layer=True, loss=None, k_predictions=1, prop_hidden=0.5, detach_c=False, random_masking=False):
        super(CNN, self).__init__()

        d = patch_size ** num_layers
        self.d = d
        self.layerwise = layerwise
        if isinstance(h, list):
            self.h = h
        else:
            self.h = [h for _ in range(num_layers)]
        self.out_dim_beta = out_dim

        self.hier = nn.Sequential(
            NonOverlappingConv1dReLU(
                input_channels, self.h[0], d // patch_size, patch_size, bias
            ),
            *[
                NonOverlappingConv1dReLU(
                    self.h[l-1], self.h[l], d // patch_size ** (l + 1), patch_size, bias
                )
                for l in range(1, num_layers)
            ],
        )
        if last_lin_layer:
            self.initialize_beta()
        else:
            self.beta = None
        self.training_layers = list(range(len(self.hier)))

        # define loss if need be
        if loss == "clapp_unsup":
            if last_lin_layer:
                raise NotImplementedError
            if self.layerwise:
                self.losses = nn.ModuleList([CLAPPUnsupervisedHalfMasking(
                    self.h[l], d // patch_size**(l+1), k_predictions=k_predictions, prop_hidden=prop_hidden,
                    detach_c=detach_c, random_masking=random_masking)
                                             for l in range(0, num_layers)])
            else:
                self.losses = CLAPPUnsupervisedHalfMasking(self.h[-1], d // (patch_size**num_layers),
                                                           k_predictions=k_predictions, prop_hidden=prop_hidden,
                                                           detach_c=detach_c, random_masking=random_masking)
        else:
            if self.layerwise:
                raise NotImplementedError("Layerwise for loss other than CLAPP not implemented.")
            self.losses = None

        self.evaluating = False

    def initialize_beta(self):
        self.beta = nn.Parameter(torch.randn(self.h[-1], self.out_dim_beta))

    def forward(self, x):
        outs = []
        if self.layerwise:
            y = x[..., :self.d]
            for mod in self.hier:
                y = mod(y.detach())
                outs.append(y)
        else:
            y = self.hier(x[..., :self.d]) # modification to look at a part of the input only if the hierarchy is not deep enough
            outs.append(y)
        y = y.mean(dim=[-1])   # last dimension (space) should already be of size 1, so equivalent to squeeze(-1)
        if self.beta is not None:
            if self.evaluating:
                y = y.detach()
            dev = y.get_device()
            if dev >= 0:
                beta = self.beta.to(dev)
            else:
                beta = self.beta
            y = y @ beta / beta.size(0)   # @ is .matmul   # does this .to(device) work? Do the gradients get properly computed?
            outs.append(y)
        return y, outs

    def compute_loss(self, outs, y, criterion=None):
        if self.layerwise:
            loss = 0
            for i, out in enumerate(outs[1]):
                if i not in self.training_layers:
                    continue
                if criterion is not None:
                    loss += criterion(out, y)
                else:
                    loss += self.losses[i](out, y)
        else:
            if criterion is not None:
                loss = criterion(outs[0], y)
            else:
                loss = self.losses(outs[0], y)
        return loss

    def eval_mode(self, on=True):
        if not on:
            raise NotImplementedError("Turning eval mode off not implemented")
        if self.beta is None:
            self.initialize_beta()
        else:
            raise ValueError("Is it ok that beta already exists??")

        self.evaluating = True
        self.layerwise = False
        self.losses = None

    def train_only_layer(self, layer):
        if layer == "all":
            self.training_layers = set(range(len(self.hier)))
        else:
            if not isinstance(layer, int):
                raise ValueError("Only 'all' or int (index of layer to train) are valid")
            self.training_layers = {layer}


class CNNLayerWise(nn.Module):    # only for patch_size = 2!!
    """
        CNN crafted to have an effective size equal to the corresponding HLCN.
        Trainable layerwise.
    """

    def __init__(self, input_channels, h, out_dim, num_layers, bias=False):
        super(CNNLayerWise, self).__init__()

        d = 2 ** num_layers
        self.d = d
        self.h = h
        self.out_dim = out_dim

        layers = []
        for l in range(num_layers):
            layers.append(NonOverlappingConv1d(h if l else input_channels, h, d // 2 ** (l + 1), bias))
            layers.append(nn.ReLU())
        self.layers = layers

        self.hier = nn.Sequential(*layers)
        self.beta = nn.Parameter(torch.randn(h, out_dim))

    def init_layerwise_(self, l):        # this allows to keep the parameters but change the layer which is trained
        device = self.beta.device
        self.hier = nn.Sequential(*self.layers[:2 * (l + 1)])
        self.beta = nn.Parameter(torch.randn(self.h, self.out_dim, device=device))

    def forward(self, x):
        if len(self.hier) == len(self.layers):
            y = self.hier(x)
        else:
            with torch.no_grad():        # this trick allows to not train everything going under no_grad()
                y = self.hier[:-2](x)
            y = self.hier[-2:](y)
        y = y.mean(dim=[-1])
        y = y @ self.beta / self.beta.size(0)
        return y