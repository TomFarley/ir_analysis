#!/usr/bin/env python

"""


Created: 
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.plotting import plot_tools

from ir_analysis.mu01.EXH01.EXH01_cd_vs_sx_profiles import shots_cd, shots_sx

from ir_analysis.compare_shots.compare_shots import compare_shots_1d_with_uda_signals

logger = logging.getLogger(__name__)

def compare_density():
    camera = 'rit'
    # signals = ['/ane/density', 'ne_bar']
    # signals = ['greenwald_density']
    signals = ['greenwald_frac']

    label = 'CD'
    # label = 'SX'

    if label == 'CD':
        shots = shots_cd
    elif label == 'SX':
        shots = shots_sx

    compare_shots_1d_with_uda_signals(camera, signals_ir=(), signals_uda=signals, pulses=shots, show=False)

    signals_str = "_".join(sig.replace('/', '') for sig in signals)
    path_fn = f'figures/{signals_str}_{label}_{len(shots)}_shots.png'
    plot_tools.save_fig(path_fn=path_fn)
    plot_tools.show_if()

if __name__ == '__main__':
    compare_density()
    pass