#!/usr/bin/env python

"""


Created: 
"""

import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from fire.scripts.scheduler_workflow import scheduler_workflow, copy_uda_netcdf_output
from fire.interfaces import uda_utils, io_basic
from fire.plotting import plot_tools

logger = logging.getLogger(__name__)

path_archive = Path('./fire_nc_output/').resolve()
fn_uda_archive_format = '{diag_tag_raw}0{shot}-alpha_{alpha:d}.nc'

def save_fire_nc_output_for_alpha_scan(shot, alphas, diag_tag_raw='rit', recompute=False):
    for alpha in alphas:
        fn_uda_archive = fn_uda_archive_format.format(shot=shot, diag_tag_raw=diag_tag_raw, alpha=int(alpha))
        path_fn_uda_archive = path_archive / fn_uda_archive
        if (not recompute) and path_fn_uda_archive.is_file():
            logger.info(f'Skipping alpha with existing uda nc archive file: {alpha}')
            continue

        outputs = scheduler_workflow(shot, camera=diag_tag_raw, alpha_user=alpha, movie_plugins_filter=['ipx'],
                                     scheduler=True)

        copy_uda_netcdf_output(outputs=outputs, path_archive=path_archive, fn_archive=fn_uda_archive, clean_netcdf=True)

def read_fire_nc_output_for_alpha_scan(shot, alphas, diag_tag_raw='rit', update_uda_scratch=True):
    uda_module, client = uda_utils.get_uda_client()

    data = {}

    for alpha in alphas:

        fn_uda_archive = fn_uda_archive_format.format(shot=shot, diag_tag_raw=diag_tag_raw, alpha=int(alpha))
        path_fn_uda_archive = path_archive / fn_uda_archive
        if not path_fn_uda_archive.is_file():
            logger.info(f'No uda nc archive file for alpha={alpha}')
            continue

        # Have to copy to scratch to uda can see the file....
        path_fn_scratch = Path('/common/uda-scratch/tfarley/file.nc').with_name(fn_uda_archive)
        if (not path_fn_scratch.is_file() or update_uda_scratch):
            io_basic.copy_file(path_fn_uda_archive, path_fn_scratch, overwrite=True, verbose=True)
            logger.debug(f'Copied nc file to {path_fn_scratch} from {path_fn_uda_archive}')

        data.setdefault(alpha, defaultdict(dict))
        
        signals = client.list_file_signals(path_fn_scratch)
        for signal in signals:
            data[alpha][signal] = client.get(signal, str(path_fn_scratch))

    return data

def plot_energy_to_divertor_curve(data, ax=None, r_cutoff = 1.06, meta_data=(), format_axes=True, show=True):
    meta_data = dict(meta_data)
    shot = meta_data.get('shot')

    fig, axes, ax_passed = plot_tools.get_fig_ax(num='power to divertor curve', axes_flatten=True, ax=ax)
    ax = axes[0]

    alphas = []
    energies = []

    for alpha, signals in data.items():
        R = signals['/ait/path0/R'].data
        mask = R < r_cutoff
        signal = '/ait/path0/cumulative_energy_vs_R'
        energy = signals['/ait/path0/cumulative_energy_vs_R']
        if isinstance(energy, dict):
            logger.info(f'Skipping data without signal {signal}')
            continue
        energy_t2_t4 = energy.data[mask][-1]
        alphas.append(alpha)
        energies.append(energy_t2_t4)

    alphas = np.array(alphas)
    alphas /= 1e6  # [W/(m$^2$•K)] -> [MW/(m$^2$•K)]
    
    ax.plot(alphas, energies, ls='-', marker='o', markersize=4, label=f'#{shot}')

    if format_axes:
        ax.set_xlabel(fr'$\alpha$ [MW/(m$^2$•K)] ')
        ax.set_ylabel(fr'$E_{{div,tot}}(R<{r_cutoff:0.2f})$ [MJ]')
        ax.ticklabel_format(axis='x', style='sci', useMathText=True)
        # ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    plot_tools.legend(ax, only_multiple_artists=False)
    plot_tools.show_if(show=show)
    pass

def plot_heat_flux_profiles(data, t_profile, r_cutoff = 1.06,  ax=None, meta_data=(), format_axes=True, show=True):
    meta_data = dict(meta_data)
    shot = meta_data.get('shot')

    fig, axes, ax_passed = plot_tools.get_fig_ax(num='power to divertor curve', axes_flatten=True, ax=ax)
    ax = axes[0]

    for alpha, signals in data.items():
        R = signals['/ait/path0/R'].data
        t = signals['/ait/path0/t'].data
        heat_flux = signals['/ait/path0/heat_flux']

        if isinstance(heat_flux, dict):
            logger.info(f'Skipping data without signal {signal}')
            continue

        heat_flux = xr.DataArray(heat_flux.data, dims=('t', 'R'), coords={'R': R, 't': t})
        heat_flux_profile = heat_flux.sel(t=t_profile, method='nearest')

        R = heat_flux_profile['R']
        R_clipped = R.where(R < r_cutoff, drop=True)
        heat_flux_profile = heat_flux_profile.where(R_clipped, drop=True)

        heat_flux_profile /= 1e6  # W -> MW

        ax.plot(R_clipped, heat_flux_profile, ls='-', marker='o', markersize=3, label=f'#{shot}, $t={t_profile:0.2f}$')

    if format_axes:
        ax.set_xlabel(fr'$R$ [m)] ')
        ax.set_ylabel(fr'$q_{{\perp}}$ [MWm$^{{-2}}$]')
        ax.ticklabel_format(axis='x', style='sci', useMathText=True)
        # ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    plot_tools.legend(ax, only_multiple_artists=False)
    plot_tools.show_if(show=show)

if __name__ == '__main__':
    shots = [
        45360, 45388  # Elmy H-mode CDC
             ]
    # alphas = [5e4, 9e4, 12e4, 30e4]
    # alphas = [3e4, 5e4, 7e4, 9e4, 12e4, 30e4]
    alphas = [1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 10.5e4, 12e4, 17e4, 24e4, 30e4, 40e4, 50e4]
    diag_tag_raw = 'rit'

    recompute = False  # Re-write uda nc files

    fig, axes, ax_passed = plot_tools.get_fig_ax(num='Power to divertor curve', ax_grid_dims=(2, 1),
                                                 figsize=(8, 16), axes_flatten=True)
    ax0 = axes[0]
    ax1 = axes[1]

    for i, shot in enumerate(shots):
        save_fire_nc_output_for_alpha_scan(shot, alphas=alphas, diag_tag_raw=diag_tag_raw, recompute=recompute)
        data = read_fire_nc_output_for_alpha_scan(shot, alphas=alphas, diag_tag_raw=diag_tag_raw, update_uda_scratch=recompute)

        meta_data = dict(diag_tag_raw=diag_tag_raw, shot=shot)

        plot_energy_to_divertor_curve(data, ax=ax0, meta_data=meta_data, show=False, format_axes=(i == 0))
        plot_heat_flux_profiles(data, t_profile=0.2, ax=ax1, meta_data=meta_data, show=False, format_axes=(i == 0))

    plot_tools.show_if(True)

    pass