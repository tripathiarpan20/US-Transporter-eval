import torch
import torch.fft
from torch import nn
import torch.nn.functional as F

#Adapted from https://github.com/phernst/pytorch_radon/tree/torch17/pytorch_radon


# constants
PI = 4*torch.ones(1, dtype=torch.double).atan()
SQRT2 = (2*torch.ones(1, dtype=torch.double)).sqrt()


def deg2rad(x, dtype=torch.float):
    return (x*PI/180).to(dtype)
    


def ramp_filter(size):
    image_n = torch.cat([
        torch.arange(1, size / 2 + 1, 2, dtype=torch.int),
        torch.arange(size / 2 - 1, 0, -2, dtype=torch.int),
    ])

    image_filter = torch.zeros(size, dtype=torch.double)
    image_filter[0] = 0.25
    image_filter[1::2] = -1 / (PI * image_n) ** 2

    fourier_filter = torch.fft.fft(image_filter)

    return 2*fourier_filter.real


class AbstractFilter(nn.Module):
    def forward(self, x):
        input_size = x.shape[2]
        projection_size_padded = \
            max(64, int(2**(2*torch.tensor(input_size)).float().log2().ceil()))
        pad_width = projection_size_padded - input_size
        padded_tensor = F.pad(x, (0, 0, 0, pad_width))
        fourier_filter = ramp_filter(padded_tensor.shape[2]).to(x.device)
        fourier_filter = self.create_filter(fourier_filter)
        fourier_filter = fourier_filter.unsqueeze(-1)
        projection_fft = torch.fft.fft(padded_tensor, dim=2)*fourier_filter
        projection = torch.fft.ifft(projection_fft, axis=2)[:, :, :input_size]
        return projection.real.to(x.dtype)

    def create_filter(self, fourier_ramp):
        raise NotImplementedError


class IdentityFilter(AbstractFilter):
    def create_filter(self, fourier_ramp):
        return torch.ones_like(fourier_ramp)


class RampFilter(AbstractFilter):
    def create_filter(self, fourier_ramp):
        return fourier_ramp


class HannFilter(AbstractFilter):
    def create_filter(self, fourier_ramp):
        n = torch.arange(0, fourier_ramp.shape[0])
        hann = (0.5 - 0.5*(2.0*PI*n/(fourier_ramp.shape[0]-1)).cos())
        hann = hann.to(fourier_ramp.device)
        return fourier_ramp*hann.roll(hann.shape[0]//2, 0)


class LearnableFilter(AbstractFilter):
    def __init__(self, filter_size):
        super(LearnableFilter, self).__init__()
        self.filter = nn.Parameter(ramp_filter(filter_size).view(-1, 1))

    def forward(self, x):
        fourier_filter = self.filter.to(x.device)
        projection = torch.fft.fft(x, dim=2) * fourier_filter
        return torch.fft.ifft(projection, dim=2).real.to(x.dtype)

    def create_filter(self, fourier_ramp):
        raise NotImplementedError
        
class Radon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True,
                 dtype=torch.float, scikit=False):
        super(Radon, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.scikit = scikit
        self.all_grids = None
        if in_size is not None:
            self.all_grids = self._create_grids(in_size)

    def forward(self, x):
        N, C, W, H = x.shape
        assert W == H

        if self.all_grids is None:
            self.all_grids = self._create_grids(W)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(
                x,
                (pad_width[0], pad_width[1], pad_width[0], pad_width[1]),
            )

        N, C, W, _ = x.shape
        out = torch.zeros(
            N, C, W, len(self.theta),
            device=x.device,
            dtype=self.dtype,
        )

        for i in range(len(self.theta)):
            rotated = F.grid_sample(
                x,
                self.all_grids[i].repeat(N, 1, 1, 1).to(x.device),
                align_corners=False,
            )
            out[..., i] = rotated.sum(2)

        return out

    def _create_grids(self, grid_size):
        if not self.circle:
            grid_size = int((SQRT2*grid_size).ceil())
        grid_shape = [1, 1, grid_size, grid_size]
        all_grids = []
        for theta in self.theta:
            theta = deg2rad(theta, self.dtype)
            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0],
                [-theta.sin(), theta.cos(), 0],
            ]], dtype=self.dtype)
            if self.scikit and grid_size % 2 == 0:
                R = self._apply_pixel_shift(R, grid_size)
            all_grids.append(F.affine_grid(R, grid_shape, align_corners=False))
        return all_grids

    def _apply_pixel_shift(self, matrix, grid_size):
        shift_fwd = torch.tensor([[
            [1, 0, -1/grid_size],
            [0, 1, -1/grid_size],
            [0, 0, 1]
        ]], dtype=self.dtype)
        shift_bwd = torch.tensor([[
            [1, 0, 1/grid_size],
            [0, 1, 1/grid_size],
            [0, 0, 1]
        ]], dtype=self.dtype)
        hom_mat = torch.cat([matrix, torch.tensor([[
                [0, 0, 1]
            ]], dtype=self.dtype)], dim=-2)
        return (shift_bwd@hom_mat@shift_fwd)[:, :2]


# TODO: something is wrong for even image sizes (probably wrong pixel shift)
class IRadon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True,
                 use_filter=RampFilter(), out_size=None, dtype=torch.float,
                 scikit=False):
        super(IRadon, self).__init__()
        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size
        self.dtype = dtype
        self.scikit = scikit
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid()
            self.all_grids = self._create_grids()
        self.filter = IdentityFilter() if use_filter is None else use_filter

    def forward(self, x):
        it_size = x.shape[2]
        ch_size = x.shape[1]

        if self.in_size is None:
            self.in_size = int((it_size/SQRT2).floor()) \
                if not self.circle else it_size
        if None in [self.ygrid, self.xgrid, self.all_grids]:
            self.ygrid, self.xgrid = self._create_yxgrid()
            self.all_grids = self._create_grids()

        x = self.filter(x)

        reco = torch.zeros(
            x.shape[0], ch_size, it_size, it_size,
            device=x.device,
            dtype=self.dtype,
        )
        for i_theta in range(len(self.theta)):
            reco += F.grid_sample(
                x,
                self.all_grids[i_theta].repeat(
                    reco.shape[0], 1, 1, 1
                ).to(x.device),
                align_corners=self.scikit,
            )

        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(
                reco,
                (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1]),
            )

        if self.circle:
            reconstruction_circle = (self.xgrid ** 2 + self.ygrid ** 2) <= (1 if self.scikit else (1-1/self.in_size))
            reconstruction_circle = reconstruction_circle.repeat(
                x.shape[0], ch_size, 1, 1,
            )
            reco[~reconstruction_circle] = 0.

        reco *= PI.to(reco.device)/(2*len(self.theta))

        if self.out_size is not None:
            pad = (self.out_size - self.in_size)//2
            reco = F.pad(reco, (pad, pad, pad, pad))

        return reco

    def _create_yxgrid(self):
        in_size = self.in_size
        if not self.circle:
            in_size = int((SQRT2*self.in_size).ceil())
        if in_size % 2 == 0 and self.scikit:
            warnings.warn("The image size is even. This leads to different "
                          "reconstructions than those obtained with "
                          "`skimage.transform.iradon` if the sinogram was "
                          "created with `skimage.trasnform.radon`.")
        unitrange = F.affine_grid(torch.eye(3)[None, :2], (1, 1, 2, in_size), align_corners=self.scikit)[0, 0, :, 0]
        return torch.meshgrid(unitrange, unitrange)

    def _xy_to_t(self, theta):
        return self.xgrid*deg2rad(theta, self.dtype).cos() - \
            self.ygrid*deg2rad(theta, self.dtype).sin()

    def _create_grids(self):
        grid_size = self.in_size
        if not self.circle:
            grid_size = int((SQRT2*self.in_size).ceil())
        all_grids = []
        theta_x = F.affine_grid(torch.eye(3)[None, :2], (1, 1, 2, len(self.theta)), align_corners=self.scikit)[0, 0, :, 0]
        for i_theta, theta in enumerate(self.theta):
            X = torch.ones([grid_size]*2, dtype=self.dtype)*theta_x[i_theta]
            Y = self._xy_to_t(theta)
            all_grids.append(torch.stack((X, Y), dim=-1).unsqueeze(0))
        return all_grids
        
        
        
#EXAMPLE
#https://colab.research.google.com/drive/1okSgnOZuDvjyGRuu1IiHXRiViQodWYeg#scrollTo=Od0cht0CVnln&line=1&uniqifier=1






