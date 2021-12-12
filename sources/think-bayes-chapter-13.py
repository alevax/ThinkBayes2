# -*- coding: utf-8 -*-
"""
@author: afpvax
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import getpass


whoamai = getpass.getuser()
current_workstation = whoamai
base_dir = '/Users/' + current_workstation + "/Workspace/ThinkBayes2/"

df = pd.read_csv( os.path.join(base_dir,'data/drp_scores.csv') , sep=";")
df.head(3)

grouped = df.groupby('Treatment')
responses = {}

for name,group in grouped:
    responses[name] = group['Response']

# getting data of the histogram
control_count, control_bins_count = np.histogram(responses["Control"], bins=20)
treated_count, treated_bins_count = np.histogram(responses["Treated"], bins=20)
# finding the PDF of the histogram using count values
control_pdf = control_count / sum(control_count)
treated_pdf = treated_count / sum(treated_count)

# using numpy np.cumsum to calculate the CDF
# We can also find using the PDF values by looping and adding
control_cdf = np.cumsum(control_pdf)
treated_cdf = np.cumsum(treated_pdf)

# # plotting PDF and CDF
# plt.plot(control_bins_count[1:], control_pdf, color="red", label="PDF")
# plt.plot(control_bins_count[1:], control_cdf, label="CDF")
# plt.legend()
#
# # plotting PDF and CDF
# plt.plot(treated_bins_count[1:], treated_pdf, color="red", label="PDF")
# plt.plot(treated_bins_count[1:], treated_cdf, label="CDF")
# plt.legend()

plt.plot(control_bins_count[1:], control_cdf, color="orange", label="Control CDF")
plt.plot(treated_bins_count[1:], treated_cdf, color="green" , label="Treatment CDF")
plt.legend()

from empiricaldist import Pmf
def make_uniform(qs, name=None, **options):
    """Make a Pmf that represents a uniform distribiton."""
    pmf = Pmf(1.0, qs, **options)
    pmf.normalize()
    if name:
        pmf.index.name = name
    return pmf

import numpy as np

qs = np.linspace(20,80,num=101)
prior_mu = make_uniform(qs, name="mean")

qs = np.linspace(5,30,num=101)
prior_sigma = make_uniform(qs,name="std")

from os.path import basename, exists
def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve
        local, _ = urlretrieve(url, filename)
        print('Downloaded ' + local)

download('https://github.com/AllenDowney/ThinkBayes2/raw/master/soln/utils.py')

from utils import make_joint
prior = make_joint(prior_mu,prior_sigma)

data = responses['Control']
data.shape

mu_mesh , sigma_mesh, data_mesh = np.meshgrid(
    prior.columns, prior.index, data)

mu_mesh.shape

from scipy.stats import norm
densities = norm(mu_mesh,sigma_mesh).pdf(data_mesh)
densities.shape

likelihood = densities.prod(axis=2)
likelihood.shape

from utils import normalize
posterior = prior * likelihood
normalize(posterior)
posterior.shape

def update_norm(prior,data):
    """Update the prior based on data."""
    mu_mesh, sigma_mesh, data_mesh = np.meshgrid(
        prior.columns, prior.index, data)
    densities = norm(mu_mesh, sigma_mesh).pdf(data_mesh)
    likelihood = densities.prod(axis=2)
    posterior = prior * likelihood
    normalize(posterior)
    return posterior

data = responses['Control']
posterior_control = update_norm(prior,data)
data = responses['Treated']
posterior_treated = update_norm(prior,data)

from utils import marginal
pmf_mean_control = marginal(posterior_control,0)
pmf_mean_treated = marginal(posterior_treated,0)

Pmf.prob_gt(pmf_mean_treated,pmf_mean_control)

from utils import plot_joint