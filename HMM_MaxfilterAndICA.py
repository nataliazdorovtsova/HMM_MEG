"""

This is a preprocessing pipeline for one RED MEG participant 
(resting-state)

"""

import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import collections
import joblib
import mne
sys.path.insert(0, '/imaging/astle/users/nz01/RED_MEG/Scripts/REDTools/')
from REDTools import preprocess

#%% CHANGE PATHS HERE!!!

resting_path = '/megdata/cbu/red/meg23_calmred03/230419'
outpath = '/imaging/astle/users/nz01/RED_MEG/Maxfiltered_ICA_clean'

#%% MAXFILTER

target_file = 'restingstate_raw.fif'

MF_settings = dict(
    max_cmd='/imaging/local/software/neuromag/bin/util/maxfilter-2.2.12',
    f=os.path.join(resting_path, target_file),
    o=os.path.join(outpath, target_file.split('restingstate')[0] + 'calm02_sss' + target_file.split('restingstate')[1]),
    lg=os.path.join(outpath, target_file.split('.')[0] + '.log'),
    trans='default',
    regularize='in',
    frame='head',
    st='10',
    cor='0.98',
    orig='0 0 45',
    inval='8',
    outval='3',
    movecomp='inter',
    bads_cmd='',
)

preprocess.maxFilt(cluster=True, **MF_settings)

#%% PLOT IT
filepath = '/imaging/astle/users/nz01/RED_MEG/Maxfiltered_ICA_clean/calm06_sss_raw.fif'
data = mne.io.read_raw_fif(filepath)
data.plot()

#%% Remove EOG and ECG and prepare for ICA

outdir = outpath
overwrite = True
preprocess.__preprocess_individual('/imaging/astle/users/nz01/RED_MEG/Maxfiltered_ICA_clean/calm06_sss_raw.fif',outdir,overwrite)

#%% ICA - make sure to change file name appropriately

data = mne.io.read_raw_fif('/imaging/astle/users/nz01/RED_MEG/Maxfiltered_ICA_clean/calm06_sss_clean_raw.fif')
man_ica = [data]

ica = mne.preprocessing.ICA(n_components=25, method='fastica').fit(data)
comps = ica.plot_components()
comps[0].savefig('/imaging/astle/users/nz01/RED_MEG/comp1.png')
comps[1].savefig('/imaging/astle/users/nz01/RED_MEG/comp2.png')

plots = ica.plot_properties(data, picks=[21])

#%% Manually exclude ICA components and save new file

data.load_data()
ica.exclude =[0]
ica.apply(data)
data.save('/imaging/astle/users/nz01/RED_MEG/Maxfiltered_ICA_clean/calm06_cleanica_sss.fif')

#%% PLOT IT
filepath = '/imaging/astle/users/nz01/RED_MEG/Maxfiltered_ICA_clean/calm06_cleanica_sss.fif'
data = mne.io.read_raw_fif(filepath)
data.plot()
