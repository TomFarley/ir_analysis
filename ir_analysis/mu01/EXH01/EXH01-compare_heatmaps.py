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

import pyuda

import fire

from ir_analysis.compare_shots.compare_shots import compare_shots_2d
from fire.plotting import plot_tools
from fire.interfaces import interfaces, io_basic, io_utils
from fire.plugins import plugins

from ir_analysis.mu01.EXH01.EXH01_cd_vs_sx_profiles import shots_cd, shots_sx

# r_t2_bounds = [0.540, 0.905]
# r_t5_bounds = [1.395, 1.745]
r_t2_bounds = [0.703, 0.90]
r_t5_bounds = [1.4, 1.75]
r_t2_t3_bounds = [0.703, 1.072]

# x_range = (0.7, 0.9)  # for R on x axis, T2
# x_range = (1.4, 1.75)  # for R on x axis, T5

def compare_heatmaps():
    machine = 'mast_u'
    camera = 'rit'

    label = 'CD'
    # label = 'SX'

    if label == 'CD':
        shots = shots_cd
        r_range = r_t2_t3_bounds
        t_range = [0, 0.95]
    elif label == 'SX':
        shots = shots_sx
        r_range = None
        t_range = [0, 0.95]

    # calcam_calib = calcam.Calibration(load_filename=str(files['calcam_calib']))

    config = interfaces.json_load(fire_paths['config'], key_paths_drop=('README',))

    paths_input = config['paths_input']['input_files']
    paths_output = {key: Path(path) for key, path in config['paths_output'].items()}

    # Load machine plugins
    machine_plugin_paths = config['paths_input']['plugins']['machine']
    machine_plugin_attrs = config['plugins']['machine']['module_attributes']
    machine_plugins, machine_plugins_info = plugins.get_compatible_plugins(machine_plugin_paths,
                                                                           attributes_required=
                                                                           machine_plugin_attrs['required'],
                                                                           attributes_optional=
                                                                           machine_plugin_attrs['optional'],
                                                                           plugins_required=machine,
                                                                           plugin_type='machine')
    machine_plugins, machine_plugins_info = machine_plugins[machine], machine_plugins_info[machine]
    fire.active_machine_plugin = (machine_plugins, machine_plugins_info)

    signals = {'heat_flux': dict()}
    # signals = {'temperature': dict()}
    robust = True
    # robust = False
    t_win = None  # Don't label time window
    # t_range = [0, 0.35]
    # t_range = [0, 0.7]

    colorbar_kwargs = {}
    # colorbar_kwargs = dict(vmin=-0.011, vmax=0.020, extend='neither')  # , extend='neither', 'both', 'max'
    compare_shots_2d(camera, signals=signals, pulses=shots, machine=machine, machine_plugins=machine_plugins,
                     t_range=t_range, r_range=r_range, t_wins=t_win, robust=robust,
                     set_ax_lims_with_ranges=True, show=False, colorbar_kwargs=colorbar_kwargs,
                     robust_percentiles=(50, 99.5))

    plot_tools.show_if(True, tight_layout=True)

if __name__ == '__main__':
    compare_heatmaps()