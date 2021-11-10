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


def heatmap_sx_sweep(pulse=44677):

    # pulse = 45411  # First shot of EXH-06 (power balance), Zref scan
    # pulse = 45414  # First shot of EXH-06 (power balance), CD -8cm Zshift, sp sweep
    pulse = 45415  # First shot of EXH-06 (power balance), SX -8cm Zshift, sp sweep
    pulse = 45416  # First shot of EXH-06 (power balance), SX -8cm Zshift, sp sweep


    camera = 'rit'
    signal_ir = 'heat_flux'
    # signal_ir = 'temperature'


    machine = 'mast_u'
    # machine = 'mast'
    meta = dict(camera=camera, pulse=pulse, machine=machine, signal=signal_ir, diag_tag_analysed=camera.upper(),
                path_label='T5')

    data = read_data_for_pulses_pickle(camera, pulse, machine)
    path_data = data[pulse][0]['path_data']
    path_data = path_data.swap_dims({'i_path0': 'R_path0'})

    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, 1), sharex=True, axes_flatten=True)

    ax = axes[0]

    # robust = False
    robust = True
    # robust_percentiles = (0, 100)
    # robust = True
    robust_percentiles = (30, 98)

    t_range = None
    r_range = None
    if pulse == 45415:
        t_range = [0.5, 0.987]
        r_range = [1.395, 1.7]
        pass
    elif pulse == 45416:
        t_range = [0.45, 1.15]
        r_range = [1.393, 1.7]
        pass

    debug_plots.debug_plot_profile_2d(data_paths=path_data, param=signal_ir, ax=ax, robust=robust,
                                      t_range=t_range, r_range=r_range, set_data_coord_lims_with_ranges=True,
                                      robust_percentiles=robust_percentiles,
                                          machine_plugins='mast_u',
                                      show=False)
    plot_tools.annotate_providence(ax, meta_data=meta)
    plot_tools.legend(ax)

    cropped_str = '-cropped' if (t_range or r_range) else ''
    times_str = "_".join([f'{t:0.3f}' for t in t_range]) if t_range else ''
    r_str = "_".join([f'{r:0.3f}' for r in r_range]) if r_range else ''

    fn = Path(f'figures/{pulse}/{signal_ir}_map-{pulse}-{machine}-t_{r_str}{cropped_str}.png').resolve()
    plot_tools.save_fig(fn, mkdir_depth=2)
    plot_tools.show_if(True, tight_layout=True)

def radial_profile():
    # pulse = 45411  # First shot of EXH-06 (power balance), Zref scan
    # pulse = 45414  # First shot of EXH-06 (power balance), CD -8cm Zshift, sp sweep
    # pulse = 45415  # First shot of EXH-06 (power balance), SX -8cm Zshift, sp sweep
    # pulse = 45416  # First shot of EXH-06 (power balance), SX -8cm Zshift, sp sweep
    pulse = 45419  # First shot of EXH-06 (power balance), CD -8cm Zshift, sp sweep
    # pulse = 45420  # First shot of EXH-06 (power balance), CD -8cm Zshift, sp sweep, with S beam for 300ms

    camera = 'rit'
    signal_ir = 'heat_flux'
    # signal_ir = 'temperature'

    machine = 'mast_u'
    # machine = 'mast'
    meta = dict(camera=camera, pulse=pulse, machine=machine, signal=signal_ir, diag_tag_analysed=camera.upper(),
                path_label='T5')

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
    t_range = [0.0, 1.2]

    t_profile = 0.5
    t_profiles = [0.9, 0.95, 1.0]

    if pulse == 45411:
        # t_profile = 0.8
        t_profile = 0.46
        # t_profile = 0.36
        t_profiles = [0.14, 0.232]
    elif pulse == 45416:
        t_profiles = [0.91, 0.96, 1.01]
        r_range = [0.7, 1.6]
    elif pulse == 45419:
        t_profiles = [0.588, 0.657, 0.736]
        r_range = [0.75, 1.0]
        t_range = [0.28, 1.2]
    elif pulse == 45420:
        t_profiles = [0.588, 0.657, 0.736]
        r_range = [0.75, 1.0]
        t_range = [0.28, 1.0]
        # r_range = [1.393, 1.6]

    cropped_str = '-cropped' if (t_range or r_range) else ''

    ax = axes[0]
    debug_plots.debug_plot_profile_2d(data_paths=path_data, param=signal_ir, ax=ax, robust=robust, mark_peak=False,
                                      t_range=t_range, r_range=r_range, set_data_coord_lims_with_ranges=True,
                                      robust_percentiles=robust_percentiles,
                                          machine_plugins='mast_u', colorbar_kwargs=dict(position='top'),
                                      show=False)
    plot_tools.annotate_providence(ax, meta_data=meta)

    ax = axes[1]
    for t_profile in t_profiles:
        profile = path_data[f'{signal_ir}_path0'].sel(t=t_profile, method='nearest')
        profile = profile.sel(R_path0=slice(*r_range)) if r_range else profile
        t_change = np.ptp(profile.data)
        label = rf'$t={t_profile:0.3f}$ s, $\Delta T = {t_change:0.2f}^\circ$C' if signal_ir == 'temperature' else (
                rf'$t={t_profile:0.3f}$ s, $\Delta q_{{\perp}}$ = {t_change:0.3f} MW'
        )
        profile.plot(ax=ax, label=label)
        color = plot_tools.get_previous_line_color(ax)
        axes[0].axhline(t_profile, ls='--', color=color)

        # plot_tools.annotate_axis(ax, rf'', loc='top_centre')
    plot_tools.legend(ax, only_multiple_artists=False)
    # ax.plot(profile, profile['R_path0'], label=f'$t={t}$ s')


    print(f'ptp = {t_change}')
    ax.set_title('')

    times_str = "_".join([f'{t:0.3f}' for t in t_profiles])
    fn = Path(f'figures/{pulse}/{signal_ir}_profile-{pulse}-{machine}-t_{times_str}{cropped_str}.png').resolve()
    fn.parent.mkdir(exist_ok=True, parents=False)
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

    plot_tools.save_fig(path_fn, verbose=True, mkdir_depth=2, image_formats=['png', 'svg'])
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
    radial_profile()
    # heatmap_sx_sweep()
    # compare_t2_t5_heat_flux()
    pass
