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
base_dir = '/Users/' + current_workstation + "/Workspace/nepc-organoids-python/"

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