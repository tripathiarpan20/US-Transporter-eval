import pandas as pd
import torchvision.io
import av
#from scipy.fftpack import fftshift, ifftshift
from phasepack.tools import rayleighmode as _rayleighmode
from phasepack.tools import lowpassfilter as _lowpassfilter
from phasepack.filtergrid import filtergrid
import time
from skimage.transform import radon, iradon, rescale

# Try and use the faster Fourier transform functions from the pyfftw module if
# available
from phasepack.tools import fft2, ifft2
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import torch
from PIL import Image
print("importing ssim")
# from pyssim.ssim.__init__ import *
print("imported ssim")

print("importing torchradon")
from torchradon import *
print("imported torchradon")

import json
from datetime import datetime
import socket
import torchvision
#from data import Dataset, Sampler
import transporter_orig
import transporter
import utils
#from generate_lus_data_wrist_radon import *
import pytorch_lightning as pl
import sys
import matplotlib.pyplot as plt


def normalise(img):
  return (img - img.min())/(img.max() - img.min() + 0.001)

def integrated_backscatter_energy(img): #img is numpy image with 1 channel
  ibs= np.cumsum(img ** 2,0)
  return ibs

def indices(i, rows):
  ret = np.zeros((rows-i+1,))
  for i in range(ret.shape[0]):
    ret[i] = ret[i] + i
  return ret

#print(indices(1,3))

def shadow(img):
  rows = img.shape[0]
  cols = img.shape[1]
  stdImg = round(rows/4)
  sh = np.zeros_like(img)

  for j in range(cols):
    for i in range(rows):
        gaussWin= np.exp(-((indices(i+1,rows))**2)/(2*(stdImg**2)))
        #print(gaussWin)
        sh[i,j] = np.sum(np.multiply(img[i:rows,j], np.transpose(gaussWin)) / np.sum(gaussWin))
        #print(sh[i,j])
        
  return sh

def analyticEstimator(img, nscale=5, minWaveLength=10, mult=2.1, sigmaOnf=0.55, k=2.,\
                 polarity=0, noiseMethod=-1):

    if img.dtype not in ['float32', 'float64']:
        img = np.float64(img)
        imgdtype = 'float64'
    else:
        imgdtype = img.dtype

    if img.ndim == 3:
        img = img.mean(2)
    rows, cols = img.shape

    epsilon = 1E-4  # used to prevent /0.
    IM = fft2(img)  # Fourier transformed image

    zeromat = np.zeros((rows, cols), dtype=imgdtype)

    # Matrix for accumulating weighted phase congruency values (energy).
    totalEnergy = zeromat.copy()

    # Matrix for accumulating filter response amplitude values.
    sumAn = zeromat.copy()

    radius, u1, u2 = filtergrid(rows, cols)

    # Get rid of the 0 radius value at the 0 frequency point (at top-left
    # corner after fftshift) so that taking the log of the radius will not
    # cause trouble.
    radius[0, 0] = 1.

    H = (1j * u1 - u2) / radius


    lp = _lowpassfilter([rows, cols], .4, 10)
    # Radius .4, 'sharpness' 10
    logGaborDenom = 2. * np.log(sigmaOnf) ** 2.

    for ss in range(nscale):
        wavelength = minWaveLength * mult ** ss
        fo = 1. / wavelength  # Centre frequency of filter

        logRadOverFo = np.log(radius / fo)
        logGabor = np.exp(-(logRadOverFo * logRadOverFo) / logGaborDenom)
        logGabor *= lp      # Apply the low-pass filter
        logGabor[0, 0] = 0.  # Undo the radius fudge

        IMF = IM * logGabor   # Frequency bandpassed image
        f = np.real(ifft2(IMF))  # Spatially bandpassed image

        # Bandpassed monogenic filtering, real part of h contains convolution
        # result with h1, imaginary part contains convolution result with h2.
        h = ifft2(IMF * H)

        # Squared amplitude of the h1 and h2 filters
        hAmp2 = h.real * h.real + h.imag * h.imag

        # Magnitude of energy
        sumAn += np.sqrt(f * f + hAmp2)

        # At the smallest scale estimate noise characteristics from the
        # distribution of the filter amplitude responses stored in sumAn. tau
        # is the Rayleigh parameter that is used to describe the distribution.
        if ss == 0:
            # Use median to estimate noise statistics
            if noiseMethod == -1:
                tau = np.median(sumAn.flatten()) / np.sqrt(np.log(4))

            # Use the mode to estimate noise statistics
            elif noiseMethod == -2:
                tau = _rayleighmode(sumAn.flatten())

        # Calculate the phase symmetry measure

        # look for 'white' and 'black' spots
        if polarity == 0:
            totalEnergy += np.abs(f) - np.sqrt(hAmp2)

        # just look for 'white' spots
        elif polarity == 1:
            totalEnergy += f - np.sqrt(hAmp2)

        # just look for 'black' spots
        elif polarity == -1:
            totalEnergy += -f - np.sqrt(hAmp2)


    if noiseMethod >= 0:
        T = noiseMethod


    else:
        totalTau = tau * (1. - (1. / mult) ** nscale) / (1. - (1. / mult))

        # Calculate mean and std dev from tau using fixed relationship
        # between these parameters and tau. See
        # <http://mathworld.wolfram.com/RayleighDistribution.html>
        EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2.)
        EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2.)

        # Noise threshold, must be >= epsilon
        T = np.maximum(EstNoiseEnergyMean + k * EstNoiseEnergySigma,
                       epsilon)
    #print(totalEnergy,'!!!!!!!!!\n')
    phaseSym = np.maximum(totalEnergy - T, 0)
    #print(phaseSym,'||||||||||||\n')
    phaseSym /= sumAn + epsilon

    #print(type(f), f.shape, f)
    #print(type(hAmp2), hAmp2.shape, hAmp2)

    LP = (1 - np.arctan2(np.sqrt(hAmp2),f))
    FS = phaseSym  #????????????
    LE = (hAmp2 + f*f)

    return LP, FS, LE  #, totalEnergy, T

def bone_prob_map(img, minwl = 10):
  ibs = normalise(integrated_backscatter_energy(img))
  LP,FS,LE = analyticEstimator(normalise(img) ** 4, minWaveLength = minwl)
  final = normalise( normalise(LP) * normalise(FS) * (1-ibs))
  meanFinal = (final*(final > 0)).mean()
  final = final * (final > 1.5*meanFinal)
  return final
"""
Assume that this class generates pairs of adjacents frames (not necessarily consecutive,
depending on 'sample_rate' variable) of US video sequences 
(with similar visual qualities, due to them being from the same video as well as same jittering applied to both........) 

"""








print("Check 1")

args = utils.ConfigDict({})
args.metric = 'mse'

def get_config():
    config = utils.ConfigDict({})
    #set by default as 10
    config.image_channels = 10
    #set by default as 10
    config.k = 10
    config.htmaplam = 0.1
    return config


def _get_model_orig(config):
    feature_encoder = transporter_orig.FeatureEncoder(config.image_channels)
    pose_regressor = transporter_orig.PoseRegressor(config.image_channels, config.k)
    refine_net = transporter_orig.RefineNet(config.image_channels)

    return transporter.Transporter(feature_encoder, pose_regressor, refine_net, std = config.htmaplam)


def _get_model(config):
    feature_encoder = transporter.FeatureEncoder(config.image_channels)
    pose_regressor = transporter.PoseRegressor(config.image_channels, config.k)
    refine_net = transporter.RefineNet(config.image_channels)

    return transporter.Transporter(feature_encoder, pose_regressor, refine_net, std = config.htmaplam)

def _get_data_loader(config):
    transform = transforms.ToTensor()
    dataset = Dataset(config.dataset_root, transform=transform)
    sampler = Sampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, sampler=sampler, pin_memory=True, num_workers=4)
    return loader

class VQVAEPerceptualLoss(torch.nn.Module):
    def __init__(self, vqvae_path = 'VQVAE_unnorm_trained.pth'):
        super(VQVAEPerceptualLoss, self).__init__()
        encoder = torch.load(vqvae_path)._encoder
        encoder.eval()
        blocks = []

        encoder._residual_stack._layers[0]._block[0].inplace = False
        encoder._residual_stack._layers[0]._block[2].inplace = False
        encoder._residual_stack._layers[1]._block[0].inplace = False
        encoder._residual_stack._layers[1]._block[2].inplace = False
        for module_name, module in encoder.named_modules():
          if module_name == '':
            continue
          if 'residual_stack' in module_name and 'block.' not in module_name:
            continue
          blocks.append(module)

        #for bl in blocks:
        #  bl.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, input, target): #(batch_size, 10, 256, 256)
        x = input
        x_ = target
        loss = 0.0
        x = x.view(-1,256,256).unsqueeze(1)
        x_ = x_.view(-1,256,256).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
            x_ = block(x_)
            loss = loss + F.mse_loss(x, x_)
        return loss



class plTransporter(pl.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.model = _get_model(config)
    self.model.train()
    if args.metric == 'mse':
        self.metric = torch.nn.MSELoss()
    elif args.metric == 'perc':
        self.metric = VQVAEPerceptualLoss(args.vq_path)
    print("Initial Hlam weights are:",self.model.hlam_weights)

  def forward(self, x1, x2):       
    return self.model(x1, x2)


class plTransporter_orig(pl.LightningModule):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.model = _get_model_orig(config)
    self.model.train()
    if args.metric == 'mse':
        self.metric = torch.nn.MSELoss()
    elif args.metric == 'perc':
        self.metric = VQVAEPerceptualLoss(args.vq_path)
        
  def forward(self, x1, x2):       
    return self.model(x1, x2)