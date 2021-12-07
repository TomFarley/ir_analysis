#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path
from cycler import cycler

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib

import pyuda
import fire
from fire.scripts.read_pickled_ir_data import read_data_for_pulses_pickle
from fire.plotting import plot_tools, debug_plots
from fire.physics import physics_parameters

logger = logging.getLogger(__name__)
logger.propagate = False

def heatmap_sx_sweep(pulse=44677):
    # pulse = 44677  # RT18 - Swept (attached?) super-x
    pulse = 44683   # RT18 - Attached T5 super-x
    # pulse = 28623   # MAST 700kA Ohmic/beam heated?
    pulse = 26505   # MAST 700kA Ohmic/beam heated?
    pulse = 26798   # MAST 400kA

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
                                      t_range=t_range, x_range=r_range, set_data_coord_lims_with_ranges=True,
                                      robust_percentiles=robust_percentiles,
                                      machine_plugins='mast_u',
                                      show=False)
    plot_tools.annotate_providence(ax, meta_data=meta)
    plot_tools.show_if(True, tight_layout=True)

def compare_heatmaps_sp_plitting(pulses):

    from fire.interfaces import interfaces
    from fire.plugins import plugins
    from fire.scripts.compare_shots import compare_shots_2d
    camera = 'rit'
    machine = 'mast_u'

    config = interfaces.json_load(fire_paths['config'], key_paths_drop=('README',))

    paths_input = config['paths_input']['input_files']
    paths_output = {key: Path(path) for key, path in config['paths_output'].items()}

    # Load machine plugins
    machine_plugin_paths = config['paths_input']['plugins']['machine']
    machine_plugin_attrs = config['plugins']['machine']['module_attributes']
    machine_plugins, machine_plugins_info = plugins.get_compatible_plugins(machine_plugin_paths,
                                            attributes_required=machine_plugin_attrs['required'],
                                            attributes_optional=machine_plugin_attrs['optional'],
                                            plugins_required=machine, plugin_type='machine')
    machine_plugins, machine_plugins_info = machine_plugins[machine], machine_plugins_info[machine]
    fire.active_machine_plugin = (machine_plugins, machine_plugins_info)

    signals = {'heat_flux': dict()}
    # signals = {'temperature': dict()}

    robust = True
    # robust = False

    t_win = None  # Don't label time window
    # t_win = [[0.19], [0.35]]

    # t_range = [0, 0.35]
    # t_range = [0.05, 0.6]
    t_range = [0.05, 0.7]  # 01-09-2021
    # t_range = None

    # r_range=r_t5_bounds
    # r_range = None
    r_range = [None, 0.85]

    # robust_percentiles = (50, 99.5)
    # robust_percentiles = (35, 99.5)
    robust_percentiles = (35, 99)
    robust_percentiles = (35, 98)

    colorbar_kwargs = {}
    # colorbar_kwargs = dict(vmin=-0.011, vmax=0.020, extend='neither')  # , extend='neither', 'both', 'max'
    compare_shots_2d(camera, signals=signals, pulses=pulses, machine=machine, machine_plugins=machine_plugins,
                     t_range=t_range, r_range=r_range, t_wins=t_win, robust=robust,
                     set_ax_lims_with_ranges=True, show=False, colorbar_kwargs=colorbar_kwargs, add_colorbar=True,
                     robust_percentiles=robust_percentiles, figsize=(22, 8))
    path_fn = f'figures/{list(signals.keys())[0]}-{np.min(pulses)}_{np.max(pulses)}.png'
    plot_tools.save_fig(path_fn)
    plot_tools.show_if(show=True)

def radial_profile_sp_splitting(pulse=44547):
    """
    Splitting times:
    44547: 0.33
    44547: N/a
    44548: Broadens, but not clearly split


    """

    camera = 'rit'
    signal_ir = 'heat_flux'
    # signal_ir = 'temperature'

    machine = 'mast_u'
    # machine = 'mast'
    meta = dict(camera=camera, pulse=pulse, machine=machine, signal=signal_ir)

    data = read_data_for_pulses_pickle(camera, pulse, machine)
    path_data = data[pulse][0]['path_data']
    path_data = path_data.swap_dims({'i_path0': 'R_path0'})

    fig, axes, ax_passed = plot_tools.get_fig_ax(num=f'sp splitting {pulse}', ax_grid_dims=(2, 1), sharex=True,
                                                 axes_flatten=True, figsize=(12, 8))

    # robust = False
    # robust_percentiles = (0, 100)
    robust = True
    robust_percentiles = (25, 97.5)

    r_range = None
    # r_range = [0.96, 1.5]
    r_range = [0.705, 0.82]

    t_range = None
    # t_range = [0.3, 0.59]
    t_range = [0.05, None]

    if pulse >= 44546:
        # t_profiles = [0.26]
        t_profiles = list(np.arange(0.12, 0.45, 0.03)) + [0.6, 0.7, 0.8, 0.9]

    cropped_str = '-cropped' if (t_range or r_range) else ''

    ax = axes[0]
    debug_plots.debug_plot_profile_2d(data_paths=path_data, param=signal_ir, ax=ax, robust=robust, mark_peak=False,
                                      t_range=t_range, x_range=r_range, set_data_coord_lims_with_ranges=True,
                                      robust_percentiles=robust_percentiles,
                                      machine_plugins='mast_u', colorbar_kwargs=dict(position='top'),
                                      show=False)
    plot_tools.annotate_providence(ax, meta_data=meta)
    for t_profile in t_profiles:
        ax.axhline(t_profile, ls='--', color='k', alpha=0.5, lw=1)

    cmap = matplotlib.cm.get_cmap('Spectral')
    custom_cycler = cycler(linestyle=['-', '--', '-.', ':'])
    ax = axes[1]
    ax.set_prop_cycle(custom_cycler)
    for i, (t_profile, ls) in enumerate(zip(t_profiles, custom_cycler())):
        color = cmap(1-i/(len(t_profiles)-1))
        profile = path_data[f'{signal_ir}_path0'].sel(t=t_profile, method='nearest')
        t_change = np.ptp(profile.data)
        profile.plot(ax=ax, label=rf'$t={t_profile:0.3f}$', color=color, **ls)
        # plot_tools.annotate_axis(ax, rf'', loc='top_centre')
    plot_tools.legend(ax, only_multiple_artists=False, fontsize=10, ncol=2)
    # ax.plot(profile, profile['R_path0'], label=f'$t={t}$ s')
    print(f'ptp = {t_change}')
    ax.set_title('')
    ax.set_xlim(*r_range)

    times_str = "_".join([f'{t:0.3f}' for t in t_profiles])
    fn = Path(f'./figures/{pulse}/{signal_ir}_profile-{pulse}-{machine}-t_{times_str}{cropped_str}.png')
    fn.parent.mkdir(exist_ok=True, parents=True)
    plot_tools.save_fig(path_fn=fn)
    plot_tools.show_if(True, tight_layout=True)



if __name__ == '__main__':
    # pulses = np.arange(44547, 44559)
    # pulses = [44547, 44548, 44550, 44551, 44554, 44555, 44556, 44558]  # missing 44552, 44553,  # 28-07-2021
    # pulses = [44776, 44777, 44778, 44779, 44781, 44782, 44783, 44784, 44785, 44786, 44787]  # 24-08-2021
    pulses = [44853, 44855, 44856, 44858, 44859]  # 01-09-2021
    compare_heatmaps_sp_plitting(pulses)

    for pulse in pulses:
        print(pulse)
        radial_profile_sp_splitting(pulse)
    pass
