#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import pyuda
import fire
from fire.scripts.read_pickled_ir_data import read_data_for_pulses_pickle
from fire.plotting import plot_tools, debug_plots

logger = logging.getLogger(__name__)
logger.propagate = False


def heatmap_sx_sweep(pulse=44677):

    camera = 'rit'
    # signal_ir = 'heat_flux'
    signal_ir = 'temperature'


    # machine = 'mast_u'
    machine = 'mast'
    meta = dict(camera=camera, pulse=pulse, machine=machine, signal=signal_ir)

    data = read_data_for_pulses_pickle(camera, pulse, machine)
    path_data = data[pulse][0]['path_data']
    path_data = path_data.swap_dims({'i_path0': 'R_path0'})

    fig, ax, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, 1), sharex=True, axes_flatten=True)

    robust = False
    robust_percentiles = (0, 100)
    # robust = True
    # robust_percentiles = (30, 98)

    t_range = None
    r_range = None
    # t_range = [0.3, 0.59]
    # r_range = [0.96, 1.5]

    debug_plots.debug_plot_profile_2d(data_paths=path_data, param=signal_ir, ax=ax, robust=robust,
                                      t_range=t_range, r_range=r_range, set_data_coord_lims_with_ranges=True,
                                      robust_percentiles=robust_percentiles,
                                          machine_plugins='mast_u',
                                      show=False)
    plot_tools.annotate_providence(ax, meta_data=meta)
    plot_tools.show_if(True, tight_layout=True)

def radial_profile():
    pulse = 44677  # RT18 - Swept (attached?) super-x
    # pulse = 44683  # RT18 - Attached T5 super-x
    # pulse = 28623   # MAST 700kA Ohmic/beam heated?
    # pulse = 26505  # MAST 700kA Ohmic/beam heated?
    # pulse = 26798  # MAST 400kA
    # pulse = 44695  # MASTU 400kA
    pulse = 44613  # MASTU 400kA

    camera = 'rit'
    # signal_ir = 'heat_flux'
    signal_ir = 'temperature'

    machine = 'mast_u'
    # machine = 'mast'
    meta = dict(camera=camera, pulse=pulse, machine=machine, signal=signal_ir)

    data = read_data_for_pulses_pickle(camera, pulse, machine)
    path_data = data[pulse][0]['path_data']
    path_data = path_data.swap_dims({'i_path0': 'R_path0'})

    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(2, 1), sharex=True, axes_flatten=True, figsize=(6, 8))

    # robust = False
    # robust_percentiles = (0, 100)
    robust = True
    robust_percentiles = (30, 98)

    r_range = None
    # r_range = [0.96, 1.5]
    # r_range = [0.75, 1.1]

    # t_range = None
    # t_range = [0.3, 0.59]
    t_range = [0.1, 0.3]

    if pulse == 44613:
        t_profile = 0.8
        t_profile = 0.46
        t_profile = 0.36
    elif pulse == 44677:
        t_profile = 0.232
        t_profiles = [0.14, 0.232]

    cropped_str = '-cropped' if (t_range or r_range) else ''

    ax = axes[0]
    debug_plots.debug_plot_profile_2d(data_paths=path_data, param=signal_ir, ax=ax, robust=robust, mark_peak=False,
                                      t_range=t_range, r_range=r_range, set_data_coord_lims_with_ranges=True,
                                      robust_percentiles=robust_percentiles,
                                          machine_plugins='mast_u', colorbar_kwargs=dict(position='top'),
                                      show=False)
    plot_tools.annotate_providence(ax, meta_data=meta)
    for t_profile in t_profiles:
        ax.axhline(t_profile, ls='--', color='k')

    ax = axes[1]
    for t_profile in t_profiles:
        profile = path_data[f'{signal_ir}_path0'].sel(t=t_profile, method='nearest')
        t_change = np.ptp(profile.data)
        profile.plot(ax=ax, label=rf'$t={t_profile:0.3f}$, $\Delta T = {t_change:0.2f}^\circ$C s')
        # plot_tools.annotate_axis(ax, rf'', loc='top_centre')
    plot_tools.legend(ax, only_multiple_artists=False)
    # ax.plot(profile, profile['R_path0'], label=f'$t={t}$ s')


    print(f'ptp = {t_change}')
    ax.set_title('')

    times_str = "_".join([f'{t:0.3f}' for t in t_profiles])
    fn = Path(f'figures/{pulse}/{signal_ir}_profile-{pulse}-{machine}-t_{times_str}{cropped_str}.png')
    fn.parent.mkdir(exist_ok=True)
    plot_tools.save_fig(path_fn=fn)
    plot_tools.show_if(True, tight_layout=True)

if __name__ == '__main__':
    radial_profile()
    # heatmap_sx_sweep()
    # compare_t2_t5_heat_flux()
    pass
