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

# James Harrison's end of EXH-01 experiments
# Conventional divertor, some sp sweeps
shots_cd = [45468,  # Low density, Greenwald frac from 0.15 to 0.25. Dr,sep increases from 500 ms. SP sweep 0.2-0.6s
            45469,  # Repeat with higher density.
            45470,  # Repeat, increase fuelling. Disrupted at 800 ms.
            45473  # Even lower Greenwald frac, ramped 0.1 to 0.6. Add SP sweep 0.7-08s Still have dr,sep increase
            ]
# Super-X
shots_sx = [45439,  # lower density than reference
            45443,  # MWI shows much reduced splitting. Satisfactory reference for density scan. sp sweep 420-480ms
            45444,  # Increase fuelling density scan - Still some way off detachment
            45446,  # Earlier fuelling
            45450,  # Density noticeably higher.
            45456,  # Repeat with D5 tweak
            45459,  # Repeat, more fuelling, tweak D5, D6
            45461,  # Repeat, more fuelling, tweak D5, D6
            45462,  # Repeat, more fuelling, tweak D5, D6
            45463,  # Repeat, more fuelling, VDE
            45464,  # Repeat, less fuelling, VDE
            45465  # Repeat, less fuelling, tweak Z-control, lower density than expected
            ]

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

def compare_t2_t5_heat_flux():
    camera = 'rit'
    signal_ir = 'heat_flux_path0'
    # pulse = 43583
    # pulse = 43610   # Initial KPI pulse
    # pulse = 43620
    # pulse = 43624
    # pulse = 43644

    # pulse = 43587

    # pulse = 43823  # strike point splitting
    # pulse = 43835  # strike point splitting

    # pulse = 44092  # Super-X plot for fulvio
    pulse = 44158  # Super-X plot for fulvio


    machine = 'mast_u'
    meta = dict(camera=camera, pulse=pulse, machine=machine, signal=signal_ir)

    plot_s_coord = True
    # plot_s_coord = False

    # align_signals = False
    # align_signals = 'peak'
    align_signals = 'zero'

    # log_y = True
    log_y = False

    plot_t2 = True
    plot_t5 = True

    simple_labels = True
    # simple_labels = False

    # t_window = 10e-3
    t_window_t2 = 0.0
    t_window_t5 = 0.0

    # t_window_t2 = 0.006
    # t_window_t5 = 0.006

    if pulse == 43610:
        t_tile2 = 0.176   # 43610 KPI
        # t_tile2 = 0.162   # 43610 KPI
        # t_tile2 = 0.261   # Strike point splitting
        t_tile5 = 0.315   # 43610 KPI
    elif pulse == 43644:
        t_tile2 = 0.140  # 43644 KPI
        t_tile5 = 0.325  # 43644 KPI
    elif pulse == 43823:
        # t_tile2 = 0.41   # Strike point splitting
        # t_tile2 = 0.44   # Strike point splitting
        t_tile2 = 0.47   # Strike point splitting
        t_tile5 = 0.55  # not on T5
    elif pulse == 43835:
        t_tile2 = 0.47   # Strike point splitting
        t_tile5 = 0.55  # not on T5
    elif pulse == 44092:
        t_tile2 = 0.2  #
        t_tile5 = 0.6  #
    elif pulse == 44158:
        t_tile2 = 0.2  #
        t_tile5 = 0.7  #
    elif False:
        t_tile2 = 0.12559
        t_tile5 = 0.3526
    elif False:
        t_tile2 = 0.115
        t_tile5 = 0.25

    r_t2_bounds = [0.540, 0.905]
    r_t5_bounds = [1.395, 1.745]

    data = read_data_for_pulses_pickle(camera, pulse, machine)
    path_data = data[pulse][0]['path_data']
    path_data = path_data.swap_dims({'i_path0': 'R_path0'})

    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, 1), sharex=True, axes_flatten=True)
    ax = axes

    r_coord = 'R_path0'
    s_coord = 's_global_path0'
    r = path_data[r_coord]
    s = path_data[s_coord]
    t = path_data['t']

    if pulse >= 43543 and pulse <= 43621:
        t_scale_factor = 0.616342 / 0.56744
    else:
        t_scale_factor = 1
    t = t * t_scale_factor
    path_data['t'] = t
    t_tile2 = t_tile2 #* t_scale_factor


    # r_t2[0] = r.values.min()
    # r_t5[1] = r.values.max()

    heat_flux = path_data[signal_ir]

    if not t_window_t2:
        profile_t2 = heat_flux.sel(t=t_tile2, method='nearest')
    else:
        t2_mask = (t >= t_tile2-t_window_t2/2) & (t <= t_tile2+t_window_t2/2)
        profile_t2 = heat_flux.sel(t=t2_mask, method='nearest').mean(dim='t')
    t2_mask = (r >= r_t2_bounds[0]) & (r <= r_t2_bounds[1])
    profile_t2 = profile_t2.sel(R_path0=t2_mask)
    r_t2 = profile_t2[r_coord]
    s_t2 = profile_t2[s_coord]

    if not t_window_t5:
        profile_t5 = heat_flux.sel(t=t_tile5, method='nearest')
    else:
        t5_mask = (t >= t_tile5-t_window_t5/2) & (t <= t_tile5+t_window_t5/2)
        profile_t5 = heat_flux.sel(t=t5_mask, method='nearest').mean(dim='t')
    t5_mask = (r >= r_t5_bounds[0]) & (r <= r_t5_bounds[1])
    profile_t5 = profile_t5.sel(R_path0=t5_mask)
    r_t5 = profile_t5[r_coord]
    s_t5 = profile_t5[s_coord]

    if align_signals == 'peak':
        r_t2 = r_t2 - r_t2[profile_t2.argmax(dim=r_coord)]
        r_t5 = r_t5 - r_t5[profile_t5.argmax()]
        s_t2 = s_t2 - s_t2[profile_t2.argmax(dim=r_coord)]
        s_t5 = s_t5 - s_t5[profile_t5.argmax()]
    if align_signals == 'zero':
        r_t2 = r_t2 - r_t2.min()
        r_t5 = r_t5 - r_t5.min()
        s_t2 = s_t2 - s_t2.min()
        s_t5 = s_t5 - s_t5.min()

    if plot_s_coord:
        if plot_t2:
            ax.plot(s_t2, profile_t2, label=rf'Tile 2 ($t={t_tile2:0.3f}$ s)')
        if plot_t5:
            ax.plot(s_t5, profile_t5, label=rf'Tile 5 ($t={t_tile5:0.3f}$ s)', ls='--')
        # ax.set_xlabel(r'$s_{global}$ [m]')
        if not simple_labels:
            ax.set_xlabel(r'$s$ [m]')
        else:
            ax.set_xlabel(r'Distance along the target [m]')
    else:
        if plot_t2:
            ax.plot(r_t2, profile_t2, label=rf'Tile 2 ($t={t_tile2:0.3f}$ s)')
        if plot_t5:
            ax.plot(r_t5, profile_t5, label=rf'Tile 5 ($t={t_tile5:0.3f}$ s)', ls='--')
        if not simple_labels:
            ax.set_xlabel(r'$R$ [m]')
        else:
            ax.set_xlabel(r'Distance along the target [m]')

    if not simple_labels:
        ax.set_ylabel(f'{heat_flux.symbol} [{heat_flux.units}]')
    else:
        ax.set_ylabel(f'Heat flux [{heat_flux.units}]')

    ax.title.set_visible(False)

    if log_y:
        ax.set_yscale('log')
        ax.set_ylim([profile_t5.min(), None])
    else:
        ax.set_ylim([0, None])


    labels_legend = None if (not simple_labels) else ['Conventional divertor', 'Super-X divertor']
    plot_tools.legend(ax, only_multiple_artists=False, loc='center right', box=False, labels=labels_legend)

    plot_tools.annotate_providence(ax, meta_data=meta, box=False)

    if align_signals:
        fn = f'{camera}_{pulse}_T2T5_aligned.png'
    else:
        fn = f'{camera}_{pulse}_T2T5_vs_R.png'

    path_fn = fire.fire_paths['user'] / 'figures' / 'heat_flux_radial_profiles' / fn

    plot_tools.save_fig(path_fn, verbose=True, mkdir_depth=3, image_formats=['png', 'svg'])
    plot_tools.show_if(True, tight_layout=True)



    # 2d heaflux map
    fig, ax, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, 1), sharex=True, axes_flatten=True)

    debug_plots.debug_plot_profile_2d(data_paths=path_data, param='heat_flux', ax=ax, robust=True,
                                          machine_plugins='mast_u', show=False)
    ax.set_ylim([0, 0.57])
    if True:
        if t_window_t2:
            if plot_t2:
                ax.axhline(y=t_tile2-t_window_t2, ls=':', color='tab:blue', lw=2)
                ax.axhline(y=t_tile2+t_window_t2, ls=':', color='tab:blue', lw=2)

            if plot_t5:
                ax.axhline(y=t_tile5 - t_window_t5, ls='--', color='tab:orange', lw=2, alpha=0.8)
                ax.axhline(y=t_tile5 + t_window_t5, ls='--', color='tab:orange', lw=2, alpha=0.8)
        else:
            if plot_t2:
                ax.axhline(y=t_tile2, ls='--', color='tab:blue', lw=2)
            if plot_t5:
                ax.axhline(y=t_tile5, ls='--', color='tab:orange', lw=2)

    ax.set_ylabel(r'$t$ [s]')

    plot_tools.show_if(True, tight_layout=True)


if __name__ == '__main__':
    # cd_profile_vs_gwf()
    sx_profile_vs_gwf()
    # heatmap_sx_sweep()
    # compare_t2_t5_heat_flux()
    pass

