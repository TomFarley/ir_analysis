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
from fire.interfaces import io_basic, io_utils, uda_utils
from fire.physics import eich

logger = logging.getLogger(__name__)

# Peter Ryan's EXH-16 shots
# The X-divertor configurations begin at >=0.55 s.
shots_all = {
    44697: {'t': [0.575, 0.75], 'fexp': 8.5, 'ngw': 0.39, 'divertor_config': 'X-divertor', 'div_con_tag': 'XDC'},
    44699: {'t': [0.55], 'fexp': 8.5, 'ngw': 0.4, 'divertor_config': 'X-divertor', 'div_con_tag': 'XDC'},
    44700: {'t': [0.55], 'fexp': 8.5, 'ngw': 0.39, 'divertor_config': 'X-divertor', 'div_con_tag': 'XDC'},
    44702: {'t': [0.625], 'fexp': 6, 'ngw': 0.38, 'divertor_config': 'X-divertor', 'div_con_tag': 'XDC'},
    44797: {'t': [0.78], 'fexp': 3, 'ngw': 0.38, 'divertor_config': 'conventional', 'div_con_tag': 'CDC'},
    44607: {'t': [0.43], 'fexp': 3, 'ngw': 0.4, 'divertor_config': 'conventional', 'div_con_tag': 'CDC'},
            }


def cd_profile_vs_gwf():

    # pulse = shots_cd[0]
    shot = 45469

    camera = 'rit'
    signal_ir = 'heat_flux'
    # signal_ir = 'temperature'

    x_param = 'R'
    # x_param = 's_global'
    x_param_path = f'{x_param}_path0'

    machine = 'mast_u'
    # machine = 'mast'
    meta = dict(camera=camera, pulse=shot, machine=machine, signal=signal_ir, diag_tag_analysed=camera.upper(),
                path_label='T5')

    data = read_data_for_pulses_pickle(camera, shot, machine)
    path_data = data[shot][0]['path_data']
    path_data = path_data.swap_dims({'i_path0': x_param_path})

    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(2, 1), sharex=False, axes_flatten=True, figsize=(6, 8))

    # robust = False
    # robust_percentiles = (0, 100)
    robust = True
    robust_percentiles = (30, 98)

    r_range = None
    # r_range = [0.96, 1.5]
    # r_range = [0.75, 1.1]

    # t_range = None
    # t_range = [0.3, 0.59]
    t_range = [0.0, 1.2]

    t_profile = 0.5
    t_profiles = [0.9, 0.95, 1.0]

    if shot == 45469:
        t_range = [0, 0.9]
        t_profiles = [0.25, 0.4, 0.5, 0.6, 0.7]
        r_range = [0.75, 0.95]
        # s_range = None
        s_range = [1.95, 2.25]
        # r_range = [0.703, 1.072]

    elif shot == 45416:
        t_profiles = [0.91, 0.96, 1.01]
        r_range = [0.7, 1.6]

    x_range = r_range if x_param == 'R' else s_range

    cropped_str = '-cropped' if (t_range or r_range) else ''

    greenwald_frac = uda_utils.get_signal_as_dataarray('greenwald_frac', shot, normalise=False)

    ax = axes[0]
    debug_plots.debug_plot_profile_2d(data_paths=path_data, param=signal_ir, coord_path=x_param_path,
                                      ax=ax, robust=robust, mark_peak=False,
                                      t_range=t_range, x_range=x_range, set_data_coord_lims_with_ranges=True,
                                      robust_percentiles=robust_percentiles,
                                      machine_plugins='mast_u', colorbar_kwargs=dict(position='top'),
                                      show=False)
    plot_tools.annotate_providence(ax, meta_data=meta, label='{machine} {pulse} {diag_tag_analysed}')

    ax = axes[1]
    for t_profile in t_profiles:
        greenwald_frac_i = float(greenwald_frac.sel({'t': t_profile}, method='nearest'))

        profile = path_data[f'{signal_ir}_path0'].sel(t=t_profile, method='nearest')
        if x_param_path == 'R_path0':
            profile = profile.sel(R_path0=slice(*r_range)) if r_range else profile
        elif x_param_path == 's_global_path0':
            profile = profile.sel(s_global_path0=slice(*s_range)) if s_range else profile

        eich_fit = eich.fit_eich_to_profile(profile[x_param_path].data, profile.data)

        ptp_change = np.ptp(profile.data)
        if signal_ir == 'temperature':
            label = rf'$t={t_profile:0.3f}$ s, $f_{{GW}}={greenwald_frac_i:0.2f}$, $\Delta T = {ptp_change:0.2f}^\circ$C'
        else:
             label = rf'$t={t_profile:0.3f}$ s, $f_{{GW}}={greenwald_frac_i:0.2f}$, $\Delta q_{{\perp}}$ = {ptp_change:0.3f} MW'

        profile.plot(ax=ax, label=label)
        eich_fit.plot(ax)

        color = plot_tools.get_previous_line_color(ax)
        axes[0].axhline(t_profile, ls='--', color=color)

        # plot_tools.annotate_axis(ax, rf'', loc='top_centre')
    plot_tools.legend(ax, only_multiple_artists=False, max_col_chars=300, fontsize=10)
    # ax.plot(profile, profile['R_path0'], label=f'$t={t}$ s')


    print(f'ptp = {ptp_change}')
    ax.set_title('')

    times_str = "_".join([f'{t:0.3f}' for t in t_profiles])
    fn = Path(f'figures/{shot}/{signal_ir}_profile-{shot}-{machine}-t_{times_str}{cropped_str}.png').resolve()
    io_utils.mkdir(fn, depth=2)
    plot_tools.save_fig(path_fn=fn)
    plot_tools.show_if(True, tight_layout=True)

def sx_profile_vs_gwf():

    # pulse = shots_cd[0]
    shot = 45464

    camera = 'rit'
    signal_ir = 'heat_flux'
    # signal_ir = 'temperature'

    x_param = 'R'
    # x_param = 's_global'
    x_param_path = f'{x_param}_path0'

    plot_eich_fit = False
    # plot_eich_fit = True

    machine = 'mast_u'
    # machine = 'mast'
    meta = dict(camera=camera, pulse=shot, machine=machine, signal=signal_ir, diag_tag_analysed=camera.upper(),
                path_label='T5')

    data = read_data_for_pulses_pickle(camera, shot, machine)
    path_data = data[shot][0]['path_data']
    path_data = path_data.swap_dims({'i_path0': x_param_path})

    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(2, 1), sharex=False, axes_flatten=True, figsize=(6, 8))

    # robust = False
    # robust_percentiles = (0, 100)
    robust = True
    robust_percentiles = (30, 98)

    r_range = None
    # r_range = [0.96, 1.5]
    # r_range = [0.75, 1.1]

    # t_range = None
    # t_range = [0.3, 0.59]
    t_range = [0.0, 1.2]

    t_profile = 0.5
    t_profiles = [0.9, 0.95, 1.0]

    if shot == 45464:
        t_range = [0.4, 0.8]
        t_profiles = [0.45, 0.52, 0.6, 0.67, 0.73]
        r_range = [1.4, 1.5]
        s_range = [1.95, 2.25]
        # s_range = None
        # r_range = [0.703, 1.072]

    elif shot == 45416:
        t_profiles = [0.91, 0.96, 1.01]
        r_range = [0.7, 1.6]

    x_range = r_range if x_param == 'R' else s_range

    cropped_str = '-cropped' if (t_range or r_range) else ''

    greenwald_frac = uda_utils.get_signal_as_dataarray('greenwald_frac', shot, normalise=False)

    ax = axes[0]
    debug_plots.debug_plot_profile_2d(data_paths=path_data, param=signal_ir, coord_path=x_param_path,
                                      ax=ax, robust=robust, mark_peak=False,
                                      t_range=t_range, x_range=x_range, set_data_coord_lims_with_ranges=True,
                                      robust_percentiles=robust_percentiles,
                                      machine_plugins='mast_u', colorbar_kwargs=dict(position='top'),
                                      show=False)
    plot_tools.annotate_providence(ax, meta_data=meta, label='{machine} {pulse} {diag_tag_analysed}')

    ax = axes[1]
    for t_profile in t_profiles:
        greenwald_frac_i = float(greenwald_frac.sel({'t': t_profile}, method='nearest'))

        profile = path_data[f'{signal_ir}_path0'].sel(t=t_profile, method='nearest')
        if x_param_path == 'R_path0':
            profile = profile.sel(R_path0=slice(*r_range)) if r_range else profile
        elif x_param_path == 's_global_path0':
            profile = profile.sel(s_global_path0=slice(*s_range)) if s_range else profile
        
        if plot_eich_fit:
            eich_fit = eich.fit_eich_to_profile(profile[x_param_path].data, profile.data)

        ptp_change = np.ptp(profile.data)
        if signal_ir == 'temperature':
            label = rf'$t={t_profile:0.3f}$ s, $f_{{GW}}={greenwald_frac_i:0.2f}$, $\Delta T = {ptp_change:0.2f}^\circ$C'
        else:
             label = rf'$t={t_profile:0.3f}$ s, $f_{{GW}}={greenwald_frac_i:0.2f}$, $\Delta q_{{\perp}}$ = {ptp_change:0.3f} MW'

        profile.plot(ax=ax, label=label)
        if plot_eich_fit:
            eich_fit.plot(ax)

        color = plot_tools.get_previous_line_color(ax)
        axes[0].axhline(t_profile, ls='--', color=color)

        # plot_tools.annotate_axis(ax, rf'', loc='top_centre')
    plot_tools.legend(ax, only_multiple_artists=False, max_col_chars=300, fontsize=10)
    # ax.plot(profile, profile['R_path0'], label=f'$t={t}$ s')


    print(f'ptp = {ptp_change}')
    ax.set_title('')

    times_str = "_".join([f'{t:0.3f}' for t in t_profiles])
    fn = Path(f'figures/{shot}/{signal_ir}_profile-{shot}-{machine}-t_{times_str}{cropped_str}.png').resolve()
    io_utils.mkdir(fn, depth=2)
    plot_tools.save_fig(path_fn=fn)
    plot_tools.show_if(True, tight_layout=True)

def compare_profiles():
    diag_tag_raw = 'rit'
    signal_ir = 'heat_flux_path0'

    machine = 'mast_u'
    meta = dict(camera=diag_tag_raw, machine=machine, signal=signal_ir)

    # plot_s_coord = True
    plot_s_coord = False

    # log_y = True
    log_y = False

    r_bounds = [0.7, 1.0]
    r_t5_bounds = [1.395, 1.745]

    r_coord = 'R_path0'
    s_coord = 's_global_path0'

    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, 1), sharex=True, axes_flatten=True)
    ax = axes[0]

    for shot, shot_params in shots_all.items():
        logger.info(f'shot: {shot}, params: {shot_params}')

        meta['pulse'] = shot

        try:
            data = read_data_for_pulses_pickle(diag_tag_raw, shot, machine)
        except Exception as e:
            logger.exception(shot)
            raise e
            continue

        t_slice = shot_params['t'][0]

        path_data = data[shot][0]['path_data']
        path_data = path_data.swap_dims({'i_path0': 'R_path0'})

        r = path_data[r_coord]
        s = path_data[s_coord]
        t = path_data['t']

        heat_flux = path_data[signal_ir]

        profile = heat_flux.sel(t=t_slice, method='nearest')

        # t2_mask = (t >= t_tile2-t_window_t2/2) & (t <= t_tile2+t_window_t2/2)
        # profile_t2 = heat_flux.sel(t=t2_mask, method='nearest').mean(dim='t')

        r_mask = (r >= r_bounds[0]) & (r <= r_bounds[1])
        profile = profile.loc[r_mask]

        ax.plot(r, profile, label=rf'{shot}, {shot_params["div_con_tag"]}, $t={t_slice:0.3f}$ s')

    ax.set_xlabel(r'$R$ [m]')
    ax.set_ylabel(f'{heat_flux.symbol} [{heat_flux.units}]')

    ax.title.set_visible(False)

    plot_tools.legend(ax, only_multiple_artists=False, loc='center right', box=False)

    plot_tools.annotate_providence(ax, meta_data=meta, box=False)

    fn = f'radial_profile-{diag_tag_raw}-{shot}'
    path_fn = Path('./figures/heat_flux_radial_profiles') / fn

    plot_tools.save_fig(path_fn, verbose=True, mkdir_depth=3, image_formats=['png', 'svg'])
    plot_tools.show_if(True, tight_layout=True)


    # 2d heaflux map
    for shot, shot_params in shots_all.items():
        data = read_data_for_pulses_pickle(diag_tag_raw, shot, machine)
        path_data = data[shot][0]['path_data']
        path_data = path_data.swap_dims({'i_path0': 'R_path0'})

        t_slice = shot_params['t'][0]

        fig, ax, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, 1), sharex=True, axes_flatten=False)

        debug_plots.debug_plot_profile_2d(data_paths=path_data, param='heat_flux', ax=ax, robust=True,
                                              machine_plugins='mast_u', show=False)
        ax.set_ylim([0, 0.57])
        ax.axhline(y=t_slice, ls=':', color='tab:blue', lw=2)

        ax.set_ylabel(r'$t$ [s]')

        plot_tools.show_if(True, tight_layout=True)


if __name__ == '__main__':
    compare_profiles()
    pass

