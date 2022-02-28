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

def read_fire_nc_output_for_alpha_scan(shot, alphas, diag_tag_raw='rit'):
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
        io_basic.copy_file(path_fn_uda_archive, path_fn_scratch, overwrite=True, verbose=True)
        logger.debug(f'Coppied nc file to {path_fn_scratch} from {path_fn_uda_archive}')

        data.setdefault(alpha, defaultdict(dict))
        
        signals = client.list_file_signals(path_fn_scratch)
        for signal in signals:
            data[alpha][signal] = client.get('/ait/path0/R', str(path_fn_scratch))

    return data

def plot_energy_to_divertor_curve(data):
    fig, axes, ax_passed = plot_tools.get_fig_ax(num='power to divertor curve', axes_flatten=True)

    alphas = []
    energies = []

    ax = axes[0]
    for alpha, signals in data.items():
        R = signals['/ait/path0/R'].data
        mask = R < 1.06
        signal = '/ait/path0/cumulative_energy_vs_R'
        energy = signals['/ait/path0/cumulative_energy_vs_R']
        if isinstance(energy, dict):
            logger.info(f'Skipping data without signal {signal}')
            continue
        energy_t2_t4 = energy.data[mask][-1]
        alphas.append(alpha)
        energies.append(energy_t2_t4)


    ax.plot(alphas, energies, label=f'{alpha}')

    ax.set_xlabel(fr'$\alpha$')
    ax.set_ylabel(fr'$E$ [MJ]')

    plot_tools.show_if()
    pass


if __name__ == '__main__':
    shot = 45360  # Elmy H-mode CDC
    alphas = [3e4, 5e4, 7e4, 9e4, 12e4, 30e4]
    diag_tag_raw = 'rit'

    # save_fire_nc_output_for_alpha_scan(shot, alphas=alphas, diag_tag_raw=diag_tag_raw, recompute=False)
    data = read_fire_nc_output_for_alpha_scan(shot, alphas=alphas, diag_tag_raw=diag_tag_raw)

    plot_energy_to_divertor_curve(data)

    pass